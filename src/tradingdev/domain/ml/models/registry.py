"""Named direction-model implementations and their configuration classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tradingdev.domain.ml.models.catboost_model import CatBoostDirectionModel
from tradingdev.domain.ml.models.lightgbm_model import LightGBMDirectionModel
from tradingdev.domain.ml.models.xgboost_model import XGBoostDirectionModel
from tradingdev.domain.ml.schemas import (
    CatBoostModelConfig,
    LightGBMModelConfig,
    XGBoostModelConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel as ConfigModel

    from tradingdev.domain.ml.base import BaseModel


@dataclass(frozen=True)
class ModelKind[ConfigT: ConfigModel]:
    """A named, audited model implementation with its config schema."""

    name: str
    description: str
    config_cls: type[ConfigT]
    model_cls: Callable[[ConfigT], BaseModel]
    framework: str

    def create(self, **params: object) -> BaseModel:
        """Validate dynamic settings with this model's paired configuration."""
        return self.model_cls(self.config_cls(**params))


class RegisteredModelKind(Protocol):
    """Read-only metadata and safe construction after config-type erasure."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def framework(self) -> str: ...

    @property
    def config_cls(self) -> type[ConfigModel]: ...

    def create(self, **params: object) -> BaseModel: ...


_REGISTRY: dict[str, RegisteredModelKind] = {}


def register[ConfigT: ConfigModel](kind: ModelKind[ConfigT]) -> None:
    """Register a model kind, rejecting duplicate names."""
    if kind.name in _REGISTRY:
        raise ValueError(f"duplicate model kind: {kind.name}")
    _REGISTRY[kind.name] = kind


register(
    ModelKind(
        name="xgboost_direction",
        description="Gradient-boosted trees (XGBoost) for directional prediction.",
        config_cls=XGBoostModelConfig,
        model_cls=XGBoostDirectionModel,
        framework="xgboost",
    )
)
register(
    ModelKind(
        name="lightgbm_direction",
        description="Gradient-boosted trees (LightGBM) for directional prediction.",
        config_cls=LightGBMModelConfig,
        model_cls=LightGBMDirectionModel,
        framework="lightgbm",
    )
)
register(
    ModelKind(
        name="catboost_direction",
        description="Gradient-boosted trees (CatBoost) for directional prediction.",
        config_cls=CatBoostModelConfig,
        model_cls=CatBoostDirectionModel,
        framework="catboost",
    )
)


def list_model_kinds() -> list[str]:
    """Return all registered model kind names, sorted for stable tool output."""
    return sorted(_REGISTRY)


def get_model_kind(name: str) -> RegisteredModelKind:
    """Return the :class:`ModelKind` for ``name``.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(name)
    return _REGISTRY[name]


def create_direction_model(name: str, **params: object) -> BaseModel:
    """Build a direction model by kind name from a uniform parameter schema."""
    return get_model_kind(name).create(**params)
