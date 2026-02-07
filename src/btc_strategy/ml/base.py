"""Abstract base class for machine learning models (placeholder)."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Base interface for ML models used in strategy development."""

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """Train the model on the provided data.

        Args:
            df: Training DataFrame with features and target.
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> "pd.Series[float]":
        """Generate predictions for the provided data.

        Args:
            df: Feature DataFrame.

        Returns:
            Series of predictions.
        """
        ...
