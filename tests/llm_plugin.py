"""Make model execution an explicit pytest choice, never an implicit expense."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("TradingDev LLM tests")
    group.addoption("--llm-provider", choices=("codex", "local"), default=None)
    group.addoption("--llm-model", default=None)
    group.addoption("--llm-base-url", default="http://localhost:11434/v1")
    group.addoption("--llm-timeout", type=float, default=600.0)
    group.addoption(
        "--llm-temperature",
        type=float,
        default=None,
        help="Local model sampling override; omitted uses the server's API defaults",
    )
    group.addoption(
        "--llm-reasoning-effort",
        default=None,
        help="Local model reasoning effort (for example low); supported by the server",
    )


def pytest_configure(config: pytest.Config) -> None:
    provider = config.getoption("llm_provider")
    effort = config.getoption("llm_reasoning_effort")
    if effort is not None:
        if provider != "local":
            raise pytest.UsageError(
                "--llm-reasoning-effort requires --llm-provider local"
            )
        if not effort.strip():
            raise pytest.UsageError("--llm-reasoning-effort must not be empty")
    temperature = config.getoption("llm_temperature")
    if temperature is not None:
        if provider != "local":
            raise pytest.UsageError("--llm-temperature requires --llm-provider local")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise pytest.UsageError(
                "--llm-temperature must be finite and between 0 and 2"
            )
    if provider is None:
        return
    timeout = config.getoption("llm_timeout")
    if not math.isfinite(timeout) or timeout <= 0:
        raise pytest.UsageError("--llm-timeout must be finite and positive")
    if provider == "local":
        if not config.getoption("llm_model"):
            raise pytest.UsageError("Local LLM tests require --llm-model")
        url = urlsplit(config.getoption("llm_base_url"))
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise pytest.UsageError("--llm-base-url must be a local loopback URL")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("llm_provider") is not None:
        return
    deselected: Sequence[pytest.Item] = [
        item for item in items if item.get_closest_marker("live_llm") is not None
    ]
    items[:] = [item for item in items if item not in deselected]
    config.hook.pytest_deselected(items=deselected)


def pytest_report_header(config: pytest.Config) -> str:
    provider = config.getoption("llm_provider")
    return (
        f"LLM tests: {provider} (explicit opt-in)"
        if provider
        else "LLM tests excluded; use scripts/check-llm.sh codex|local"
    )
