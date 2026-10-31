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
        raise ValueError(
            f"{label} must be a positive integer (a built-in int of at least 1), "
            f"got {value!r} of type {type(value).__name__}"
        )
    return value


def seed_value(value: object) -> int | None:
    """Return a bootstrap ``seed`` if it is ``None`` or a built-in int >= 0.

    The rule for counts applies, with zero allowed: NumPy would accept
    ``True`` as seed 1 and reject ``1.5`` or ``"1"`` with a ``TypeError``.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "seed must be None or a non-negative built-in int, "
            f"got {value!r} of type {type(value).__name__}"
        )
    return value


def is_real(value: object) -> bool:
    """Whether ``value`` is a real number and not a boolean.

    ``bool`` is a ``numbers.Real``; ``np.bool_`` is not, but NumPy converts
    it to 1.0 all the same, so both are excluded explicitly.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _plain_numbers(values: object) -> bool:
    """Whether every element of a flat sequence has a plain numeric type.

    Collecting the element types is far cheaper than a ``numbers.Real``
    check of every element. Only Python and NumPy integers and floats
    count: ``bool`` and ``np.bool_`` are excluded, and so is anything else
    NumPy would have converted, such as a 0-d array.
    """
    types = set(map(type, values))  # type: ignore[call-overload]
    return all(
        issubclass(kind, int | float | np.integer | np.floating)
        and not issubclass(kind, bool | np.bool_)
        for kind in types
    )


def real_array(values: object, label: str) -> npt.NDArray[np.float64]:
    """Return ``values`` as a float array if every element is a real number.

    Integer and floating dtypes are accepted. Booleans, strings and other
    objects are rejected even where NumPy could convert them, so ``True`` is
    not read as 1.0 and ``"0.01"`` not as a return of one percent. Object
    arrays, such as nullable pandas columns, are checked element by element.
    So is a plain sequence unless it is flat and every element is a Python
    or NumPy integer or float: NumPy promotes ``[100.0, True]`` to floats
    before any dtype could show the boolean. Finiteness and shape are left
    to the caller.
    """
    array = np.asarray(values)
    kind = array.dtype.kind
    if getattr(values, "dtype", None) is None and not (
        kind in "iuf" and array.ndim == 1 and _plain_numbers(values)
    ):
        # A sequence NumPy could not keep as plain numbers, or one it may
        # have promoted from booleans or 0-d arrays: judge each element, so
        # the message names the first bad one.
        array = np.asarray(values, dtype=object)
        kind = "O"
    if kind == "b":
        raise ValueError(f"{label} must be numbers, not booleans")
    if kind == "O":
        for element in array.ravel().tolist():
            if not is_real(element):
                raise ValueError(f"{label} must be numbers, got {element!r}")
    elif kind not in "iuf":
        raise ValueError(f"{label} must be numbers, got dtype {array.dtype}")
    return array.astype(np.float64)
