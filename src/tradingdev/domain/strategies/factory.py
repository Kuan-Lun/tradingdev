"""Strategy lifetime operations needed by domain contract checks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine
    from tradingdev.domain.backtest.schemas import ParallelConfig
    from tradingdev.domain.strategies.base import BaseStrategy
    from tradingdev.domain.strategies.execution import StrategyExecution


class StrategyFactoryPort(Protocol):
    def open_from_config(
        self,
        raw_config: dict[str, Any],
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None = None,
    ) -> AbstractContextManager[BaseStrategy]: ...

    def open_from_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        engine: BaseBacktestEngine | None,
        parallel_config: ParallelConfig | None = None,
        *,
        parameter_overrides: dict[str, Any] | None = None,
    ) -> AbstractContextManager[BaseStrategy]: ...

    def override_execution(
        self,
        strategy_cfg: dict[str, Any],
        execution: StrategyExecution,
        parameters: dict[str, Any],
    ) -> StrategyExecution: ...
