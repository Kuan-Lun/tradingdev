"""Shared volatility estimators for the bundled GLFT strategies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import numpy.typing as npt


def compute_glft_volatility(
    close: npt.NDArray[np.floating[Any]],
    high: npt.NDArray[np.floating[Any]],
    low: npt.NDArray[np.floating[Any]],
    *,
    vol_type: Literal["realized", "parkinson", "implied"],
    window: int,
    dvol: npt.NDArray[np.floating[Any]] | None = None,
) -> npt.NDArray[np.float64]:
    """Estimate sigma, preserving missing values and the one-minute DVOL scale."""
    if vol_type == "implied":
        if dvol is None:
            raise ValueError("dvol array is required when vol_type is 'implied'")
        # Annualized percent to one-minute sigma (365.25 * 24 * 60).
        sigma = dvol / 100.0 / np.sqrt(525_960.0)
    elif vol_type == "parkinson":
        log_hl = np.log(np.maximum(high, 1e-10) / np.maximum(low, 1e-10))
        parkinson_var = log_hl**2 / (4.0 * math.log(2))
        sigma = np.sqrt(
            pd.Series(parkinson_var).rolling(window, min_periods=1).mean().values
        )
    else:
        log_ret = np.diff(np.log(np.maximum(close, 1e-10)))
        log_ret = np.concatenate([[0.0], log_ret])
        sigma = pd.Series(log_ret).rolling(window, min_periods=1).std().values

    # Unknown volatility remains missing; only nonpositive observations are floored.
    return np.asarray(np.where(sigma <= 0, 1e-10, sigma), dtype=np.float64)
