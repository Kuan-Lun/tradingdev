"""Input checks shared by the statistics modules.

Annotations describe what a public function expects, but nothing enforces
them at runtime: a ``pd.Series`` or ``npt.ArrayLike`` argument may hold
booleans, numeric strings or plain integers. Every check here narrows such
input to the declared type or raises ``ValueError`` naming the argument; none
converts a value of the wrong kind, because a silent conversion is how a
wrong input becomes a plausible but wrong answer.
"""

from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt


def positive_int(value: object, label: str) -> int:
    """Return ``value`` if it is a built-in ``int`` of at least one.

    ``bool`` is an ``int`` subclass and NumPy integers are not ``int``, so
    both are rejected. This is the rule the annotations already state: mypy
    rejects ``True`` and ``np.int64(5)`` for an ``int`` parameter as well.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def is_real(value: object) -> bool:
    """Whether ``value`` is a real number and not a boolean.

    ``bool`` is a ``numbers.Real``; ``np.bool_`` is not, but NumPy converts
    it to 1.0 all the same, so both are excluded explicitly.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def real_array(values: object, label: str) -> npt.NDArray[np.float64]:
    """Return ``values`` as a float array if every element is a real number.

    Integer and floating dtypes are accepted. Booleans, strings and other
    objects are rejected even where NumPy could convert them, so ``True`` is
    not read as 1.0 and ``"0.01"`` not as a return of one percent. Object
    arrays, such as nullable pandas columns, and plain Python sequences are
    checked element by element: NumPy promotes ``[100.0, True]`` to floats
    before any dtype could show the boolean. Finiteness and shape are left to
    the caller.
    """
    # Only input that already has a dtype can be judged by it.
    has_dtype = getattr(values, "dtype", None) is not None
    array = np.asarray(values) if has_dtype else np.asarray(values, dtype=object)
    kind = array.dtype.kind
    if kind == "b":
        raise ValueError(f"{label} must be numbers, not booleans")
    if kind == "O":
        for element in array.ravel().tolist():
            if not is_real(element):
                raise ValueError(f"{label} must be numbers, got {element!r}")
    elif kind not in "iuf":
        raise ValueError(f"{label} must be numbers, got dtype {array.dtype}")
    return array.astype(np.float64)
