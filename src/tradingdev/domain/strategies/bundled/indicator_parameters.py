"""TA-Lib moving-average period constraints for bundled strategy configs."""

from typing import Annotated

from pydantic import AfterValidator, Field


def _validate_trend_ema_period(value: int) -> int:
    if value == 1:
        msg = "trend EMA period must be 0 (disabled) or between 2 and 100000"
        raise ValueError(msg)
    return value


MovingAveragePeriod = Annotated[int, Field(strict=True, ge=2, le=100_000)]
TrendEMAPeriod = Annotated[
    int,
    Field(strict=True, ge=0, le=100_000),
    AfterValidator(_validate_trend_ema_period),
]
