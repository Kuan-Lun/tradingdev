"""Domain schemas for ML models."""

from pydantic import BaseModel


class XGBoostModelConfig(BaseModel):
    """XGBoost model hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 10
    random_state: int = 42
    n_jobs: int | None = None
