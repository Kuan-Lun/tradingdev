"""Abstract base class for machine learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseModel(ABC):
    """Base interface for ML models used in strategies."""

    @abstractmethod
    def train(
        self,
        df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> None:
        """Train the model on the provided data."""
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> pd.Series[float]:
        """Generate predictions for the provided data."""
        ...

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return class probabilities for each row."""
        ...
