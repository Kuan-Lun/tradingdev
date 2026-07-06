"""Runtime signal-contract checks shared by validate and dry-run phases."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.validator import diagnostic

if TYPE_CHECKING:
    from tradingdev.domain.strategies.schemas import (
        StrategyDiagnostic,
        StrategyMetadata,
    )

VALIDATE_FIXTURE_ROWS = 80
DRY_RUN_FIXTURE_ROWS = 240


class SignalContractChecker:
    """Execute a strategy on synthetic data and verify the signal contract.

    The contract is one gate used at two depths: validation runs a short
    fixture, dry-run a longer one. Both must agree on the rules checked here
    (return type, no input mutation, signal column limited to -1/0/1).
    """

    def __init__(self, loader: StrategyLoader | None = None) -> None:
        self._loader = loader or StrategyLoader()

    def check(
        self,
        metadata: StrategyMetadata,
        *,
        fixture_rows: int,
    ) -> dict[str, Any]:
        """Return contract diagnostics and signal analysis for a strategy."""
        diagnostics: list[StrategyDiagnostic] = []
        try:
            strategy_cfg = {
                "id": metadata.strategy_id,
                "source_path": metadata.source_path,
                "class_name": metadata.class_name,
            }
            cls = self._loader.load_class(strategy_cfg)
            strategy = self._instantiate(cls, metadata)
            if not isinstance(strategy, BaseStrategy):
                diagnostics.append(
                    _contract_diagnostic(
                        code="base_strategy_inheritance",
                        message="strategy must inherit BaseStrategy",
                        fix="Make the generated class inherit from BaseStrategy.",
                    )
                )
                return {"diagnostics": diagnostics}
            df = _fixture_df(fixture_rows)
            before = df.copy(deep=True)
            result = strategy.generate_signals(df)
            if not isinstance(result, pd.DataFrame):
                diagnostics.append(
                    _contract_diagnostic(
                        code="signals_not_dataframe",
                        message="generate_signals must return DataFrame",
                        fix="Return the copied DataFrame with a signal column.",
                    )
                )
                return {"diagnostics": diagnostics}
            if not df.equals(before):
                diagnostics.append(
                    _contract_diagnostic(
                        code="input_mutated",
                        message="generate_signals must not mutate input",
                        fix="Start with result = df.copy() and mutate result only.",
                    )
                )
            if "signal" not in result.columns:
                diagnostics.append(
                    _contract_diagnostic(
                        code="missing_signal_column",
                        message="result must include signal column",
                        fix="Add result['signal'] with values -1, 0, or 1.",
                    )
                )
                return {"diagnostics": diagnostics}
            signals = result["signal"]
            values = set(signals.dropna().unique())
            if not values.issubset({-1, 0, 1}):
                diagnostics.append(
                    _contract_diagnostic(
                        code="invalid_signal_values",
                        message="signal values must be limited to -1, 0, and 1",
                        fix=(
                            "Map all generated signals to the project convention: "
                            "-1, 0, 1."
                        ),
                    )
                )
            return {
                "diagnostics": diagnostics,
                "signal_analysis": _signal_analysis(result),
            }
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                _contract_diagnostic(
                    code="contract_execution_error",
                    message=str(exc),
                    fix="Fix the class name, constructor, imports, or signal logic.",
                )
            )
            return {"diagnostics": diagnostics}

    def _instantiate(
        self,
        cls: type[BaseStrategy],
        metadata: StrategyMetadata,
    ) -> BaseStrategy:
        config_path = Path(metadata.config_path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        strategy_cfg = raw.get("strategy", {}) if isinstance(raw, dict) else {}
        params = (
            strategy_cfg.get("parameters", {}) if isinstance(strategy_cfg, dict) else {}
        )
        params = params if isinstance(params, dict) else {}

        signature = inspect.signature(cls)
        kwargs: dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if name == "backtest_engine":
                kwargs[name] = None
            elif name in params:
                kwargs[name] = params[name]
            elif parameter.default is inspect.Parameter.empty:
                msg = f"Constructor parameter {name!r} has no default or YAML value"
                raise TypeError(msg)
        return cls(**kwargs)


def _contract_diagnostic(*, code: str, message: str, fix: str) -> StrategyDiagnostic:
    return diagnostic(code=code, phase="contract", message=message, fix=fix)


def _fixture_df(rows: int) -> pd.DataFrame:
    close: list[float] = []
    price = 100.0
    for i in range(rows):
        if rows <= VALIDATE_FIXTURE_ROWS:
            price += 0.1
        elif i < rows // 3:
            price += 0.12
        elif i < (rows * 2) // 3:
            price -= 0.08
        else:
            price += 0.18 if i % 2 == 0 else -0.11
        close.append(price)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": [value - 0.05 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [1000.0 + (i % 24) * 10.0 for i in range(rows)],
        }
    )


def _signal_analysis(result: pd.DataFrame) -> dict[str, Any]:
    signals = result["signal"]
    distribution = {
        str(key): int(value)
        for key, value in signals.value_counts(dropna=False).to_dict().items()
    }
    transitions = int(signals.fillna(0).ne(signals.fillna(0).shift()).sum() - 1)
    active = int(signals.isin([-1, 1]).sum())
    return {
        "rows": int(len(result)),
        "signal_distribution": distribution,
        "nan_count": int(signals.isna().sum()),
        "transition_count": max(transitions, 0),
        "active_signal_ratio": active / max(len(result), 1),
        "first_timestamp": (
            str(result["timestamp"].iloc[0])
            if "timestamp" in result.columns and not result.empty
            else None
        ),
        "last_timestamp": (
            str(result["timestamp"].iloc[-1])
            if "timestamp" in result.columns and not result.empty
            else None
        ),
    }
