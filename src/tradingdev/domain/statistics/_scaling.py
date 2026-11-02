"""Exact power-of-two rescaling for values at extreme magnitudes or spreads.

Statistics built from deviations (variances, higher moments, block length
estimates) overflow or underflow long before the values themselves do: a
fourth moment overflows once deviations pass about ``1e77``, and squared
deviations underflow when the spread is tiny even if the values are not.
So the decision looks at two numbers, the largest magnitude and the spread
(``max - min``). When both lie inside ``[2**-60, 2**60]`` the values are
returned unchanged. Otherwise they are multiplied by the power of two that
brings the spread into ``[0.5, 1)``, or the magnitude when there is no
finite, positive spread to use.

Multiplying by a power of two is exact for every value that stays a normal
float, and floating-point sums, products, quotients and square roots commute
with it. A statistic that is invariant to a positive scale therefore gives
bit-for-bit the same result on rescaled values whenever the unscaled
computation did not overflow or underflow, and a finite one where it did.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

_LOWER = 2.0**-60
_UPPER = 2.0**60


def scale_exponent(values: npt.NDArray[np.float64]) -> int:
    """Binary exponent ``e`` such that ``values * 2**-e`` is safe to use.

    Returns 0 when the values need no rescaling: empty, all zero, or with
    both their largest magnitude and their spread inside ``[2**-60, 2**60]``.
    ``values`` must be finite.
    """
    if values.size == 0:
        return 0
    peak = float(np.abs(values).max())
    if peak == 0.0:
        return 0
    # Python floats give inf for an overflowing spread without a warning.
    spread = float(values.max()) - float(values.min())
    if spread >= _LOWER and peak <= _UPPER:
        return 0
    if 0.0 < spread < math.inf:
        return math.frexp(spread)[1]
    # Constant values, or a spread that overflowed because the values reach
    # opposite ends of the float range: scale the magnitude below one, which
    # leaves the spread below two.
    return math.frexp(peak)[1]


def rescaled(
    values: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int]:
    """``values * 2**-e`` and ``e``, with ``e`` from :func:`scale_exponent`."""
    exponent = scale_exponent(values)
    if exponent == 0:
        return values, 0
    return np.ldexp(values, -exponent), exponent


def rescaled_columns(columns: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Each column of a 2-D array rescaled on its own, as by :func:`rescaled`.

    For statistics computed per column that are invariant to each column's
    scale, such as a block length estimate.
    """
    exponents = [scale_exponent(columns[:, k]) for k in range(columns.shape[1])]
    if not any(exponents):
        return columns
    # int32: NumPy's ldexp takes a C int or long exponent, and long is 32
    # bits on Windows; binary exponents of doubles fit either way.
    return np.ldexp(columns, -np.array(exponents, dtype=np.int32))
