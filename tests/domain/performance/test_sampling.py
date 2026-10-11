"""Frequency eligibility comes from declared bars, never elapsed timestamp gaps."""

import pytest

from tradingdev.domain.performance.sampling import daily_observation_unavailable_reason


@pytest.mark.parametrize(
    "frequency",
    ["1m", "1min", "60m", "1440m", "1T", "1h", "24h", "1d", "D", "86400s", "1ms"],
)
def test_daily_or_finer_frequency_supports_daily_observations(frequency: str) -> None:
    assert daily_observation_unavailable_reason(frequency) is None


@pytest.mark.parametrize(
    "frequency",
    ["1w", "1wk", "W", "1M", "1mo", "MS", "ME", "3d", "7D", "25h", "1441m"],
)
def test_coarse_bars_cannot_supply_daily_observations(frequency: str) -> None:
    assert (
        daily_observation_unavailable_reason(frequency) == "unsupported_daily_sampling"
    )


@pytest.mark.parametrize(
    "frequency",
    [
        "",
        "unknown",
        "0d",
        "0M",
        "-1d",
        "1.5h",
        "1m2s",
        "inf",
        "1",
        "1x",
        "9" * 5000 + "m",
    ],
)
def test_unknown_or_invalid_frequency_does_not_claim_daily_observations(
    frequency: str,
) -> None:
    assert daily_observation_unavailable_reason(frequency) == "unknown_bar_frequency"
