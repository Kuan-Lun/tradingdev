"""Execution-local random generators without modifying process-global RNG state."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class _Randomness:
    seed: int | None
    python: Random
    numpy: np.random.Generator


_CURRENT: ContextVar[_Randomness | None] = ContextVar(
    "execution_randomness", default=None
)


@contextmanager
def execution_randomness(seed: int | None) -> Generator[None]:
    """Start fresh streams for this execution and restore the enclosing context.

    Each execution (including each parallel trial) must open its own scope.
    Arbitrary global RNG calls and third-party model RNGs are not affected.
    """
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise ValueError("random_seed must be null or an integer from 0 to 2**32-1")
    token = _CURRENT.set(_Randomness(seed, Random(seed), np.random.default_rng(seed)))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _current() -> _Randomness:
    context = _CURRENT.get()
    if context is None:
        raise RuntimeError("Random generators require an active strategy execution")
    return context


def get_random() -> Random:
    """Return the current execution's Python random generator."""
    return _current().python


def get_numpy_rng() -> np.random.Generator:
    """Return the current execution's NumPy generator."""
    return _current().numpy


def get_seed() -> int | None:
    """Return the run seed for explicit third-party random_state parameters."""
    return _current().seed
