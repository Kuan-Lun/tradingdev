"""Run-local random streams remain reproducible without global side effects."""

from __future__ import annotations

import asyncio
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from tradingdev.domain.randomness import (
    execution_randomness,
    get_numpy_rng,
    get_random,
    get_seed,
)


def _draw() -> tuple[float, float]:
    return get_random().random(), float(get_numpy_rng().random())


def test_seeded_streams_replay_without_changing_global_random_state() -> None:
    python_state = random.getstate()
    numpy_state = pickle.dumps(np.random.get_state())
    with execution_randomness(42):
        assert get_seed() == 42
        expected = [_draw() for _ in range(4)]
    with execution_randomness(42):
        assert [_draw() for _ in range(4)] == expected
    with execution_randomness(7):
        assert [_draw() for _ in range(4)] != expected
    assert random.getstate() == python_state
    assert pickle.dumps(np.random.get_state()) == numpy_state
    with pytest.raises(RuntimeError, match="active strategy execution"):
        get_seed()


def test_nested_exception_restores_outer_stream_and_unseeded_scope_is_independent() -> (
    None
):
    with execution_randomness(42):
        expected = [_draw(), _draw()]
    with execution_randomness(42):
        assert _draw() == expected[0]
        with (
            pytest.raises(ValueError, match="fixture failure"),
            execution_randomness(7),
        ):
            _draw()
            raise ValueError("fixture failure")
        assert get_seed() == 42
        assert _draw() == expected[1]
        outer = get_numpy_rng()
        with execution_randomness(None):
            assert get_seed() is None
            assert get_numpy_rng() is not outer


def test_threads_have_independent_streams_despite_interleaving() -> None:
    barrier = Barrier(2)

    def run(seed: int) -> list[tuple[float, float]]:
        with execution_randomness(seed):
            values = [_draw()]
            barrier.wait(timeout=5)
            values.append(_draw())
            return values

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(run, (42, 42))
    with execution_randomness(42):
        assert first == second == [_draw(), _draw()]


def test_async_executions_have_independent_streams() -> None:
    async def run(seed: int) -> list[tuple[float, float]]:
        with execution_randomness(seed):
            values = [_draw()]
            await asyncio.sleep(0)
            values.append(_draw())
            return values

    async def exercise() -> None:
        first, second = await asyncio.gather(run(42), run(42))
        with execution_randomness(42):
            assert first == second == [_draw(), _draw()]

    asyncio.run(exercise())


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "42"])
def test_invalid_seed_cannot_silently_change_its_meaning(seed: object) -> None:
    with pytest.raises(ValueError, match="random_seed"), execution_randomness(seed):  # type: ignore[arg-type]
        pytest.fail("Invalid seed must not enter execution")
