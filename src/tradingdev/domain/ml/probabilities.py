"""Class probability validation shared by risk and direction consumers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_probabilities(
    probabilities: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Require one normalized, finite class vector per requested source row.

    A malformed model response is an error, never an unavailable feature row.
    Class labels retain their model meaning; no direction names are imposed.
    """
    if (
        not probabilities.index.equals(features.index)
        or not probabilities.columns.is_unique
        or probabilities.shape[1] == 0
    ):
        raise RuntimeError(
            "Model probabilities must match feature rows and unique classes"
        )
    values = probabilities.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise RuntimeError(
            "Model probabilities must be finite normalized values in [0, 1]"
        )
    return probabilities.astype(np.float64)


def class_probability(probabilities: pd.DataFrame, label: int) -> pd.Series:
    """Read a class; a scored distribution assigns zero to an absent class."""
    if label in probabilities.columns:
        return probabilities[label]
    return pd.Series(0.0, index=probabilities.index, dtype=np.float64)
