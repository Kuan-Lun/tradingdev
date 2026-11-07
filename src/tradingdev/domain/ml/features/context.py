"""Feature-owned history requirements consumed by rolling geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import pandas as pd


class FeatureTransformer(Protocol):
    """A labelled feature pipeline with explicit prediction requirements."""

    @property
    def minimum_context_rows(self) -> int: ...

    @property
    def context_rows(self) -> int: ...

    @property
    def target_classes(self) -> tuple[int, ...]: ...

    def transform(
        self, df: pd.DataFrame, *, include_target: bool = True
    ) -> pd.DataFrame: ...


@dataclass(frozen=True)
class FeatureContext:
    """Rows including the scored bar; preferred history is not a hard minimum."""

    minimum_rows: int
    preferred_rows: int

    def __post_init__(self) -> None:
        if self.minimum_rows < 1 or self.preferred_rows < self.minimum_rows:
            raise ValueError("Feature context requires preferred >= minimum >= 1")

    @classmethod
    def for_features(cls, features: FeatureTransformer) -> FeatureContext:
        return cls(features.minimum_context_rows, features.context_rows)
