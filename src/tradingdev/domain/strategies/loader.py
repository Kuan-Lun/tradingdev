"""Strategy loading for bundled and generated strategies."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from tradingdev.domain.backtest.schemas import ParallelConfig
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.bundled.glft_ml_strategy.config import (
    GLFTMLStrategyConfig,
)
from tradingdev.domain.strategies.bundled.glft_strategy.config import (
    GLFTStrategyConfig,
)
from tradingdev.domain.strategies.bundled.kd_strategy.config import (
    KDFitConfig,
    KDStrategyConfig,
)
from tradingdev.domain.strategies.bundled.quantile_strategy.config import (
    QuantileStrategyConfig,
)
from tradingdev.domain.strategies.bundled.safety_volume_strategy.config import (
    SafetyVolumeStrategyConfig,
)
from tradingdev.domain.strategies.bundled.xgboost_strategy.config import (
    XGBoostStrategyConfig,
)

if TYPE_CHECKING:
    from types import ModuleType

    from tradingdev.domain.backtest.base_engine import BaseBacktestEngine


@dataclass(frozen=True)
class StrategyModuleSpec:
    """Resolved strategy module metadata."""

    strategy_id: str
    source_path: Path
    class_name: str
    bundled: bool


class StrategyLoader:
    """Load bundled or workspace-generated strategies through one contract."""

    def __init__(self, *, workspace_root: Path | None = None) -> None:
        self._workspace_root = workspace_root or Path("workspace").resolve()
        self._generated_root = self._workspace_root / "generated_strategies"

    def create_from_config(
        self,
        raw_config: dict[str, Any],
        engine: BaseBacktestEngine,
        parallel_config: ParallelConfig | None = None,
    ) -> BaseStrategy:
        """Build a strategy instance from a parsed YAML configuration."""
        strategy_cfg = raw_config["strategy"]
        if not isinstance(strategy_cfg, dict):
            msg = "strategy config must be a mapping"
            raise ValueError(msg)

        strategy_id = self._required_strategy_string(strategy_cfg, "id")
        params = strategy_cfg.get("parameters", {})
        if not isinstance(params, dict):
            msg = "strategy.parameters must be a mapping"
            raise ValueError(msg)

        p_cfg = parallel_config or ParallelConfig()

        if strategy_id in {"kd_strategy", "kd_crossover"}:
            fit_config: KDFitConfig | None = None
            if "fit" in strategy_cfg:
                fit_config = KDFitConfig(**strategy_cfg["fit"])
            from tradingdev.domain.strategies.bundled.kd_strategy.strategy import (
                KDStrategy,
            )

            return KDStrategy(
                config=KDStrategyConfig(**params),
                fit_config=fit_config,
                backtest_engine=engine,
                parallel_config=p_cfg,
            )

        if strategy_id in {"xgboost_strategy", "xgboost_direction"}:
            from tradingdev.domain.strategies.bundled.xgboost_strategy.strategy import (
                XGBoostStrategy,
            )

            return XGBoostStrategy(
                config=XGBoostStrategyConfig(**params),
                backtest_engine=engine,
            )

        if strategy_id in {"safety_volume_strategy", "safety_first_volume"}:
            from tradingdev.domain.strategies.bundled.safety_volume_strategy import (
                strategy as safety_volume_module,
            )

            return safety_volume_module.SafetyVolumeStrategy(
                config=SafetyVolumeStrategyConfig(**params),
                backtest_engine=engine,
            )

        if strategy_id in {"glft_strategy", "glft_market_making"}:
            from tradingdev.domain.strategies.bundled.glft_strategy.strategy import (
                GLFTStrategy,
            )

            return GLFTStrategy(
                config=GLFTStrategyConfig(**params),
                backtest_engine=engine,
                parallel_config=p_cfg,
            )

        if strategy_id in {"quantile_strategy", "quantile_volume"}:
            from tradingdev.domain.strategies.bundled.quantile_strategy import (
                strategy as quantile_module,
            )

            return quantile_module.QuantileStrategy(
                config=QuantileStrategyConfig(**params),
                backtest_engine=engine,
                parallel_config=p_cfg,
            )

        if strategy_id in {"glft_ml_strategy", "glft_ml"}:
            from tradingdev.domain.strategies.bundled.glft_ml_strategy.strategy import (
                GLFTMLStrategy,
            )

            return GLFTMLStrategy(
                config=GLFTMLStrategyConfig(**params),
                backtest_engine=engine,
                parallel_config=p_cfg,
            )

        return self._create_generated(strategy_cfg, engine)

    def load_class(self, strategy_cfg: dict[str, Any]) -> type[BaseStrategy]:
        """Resolve a strategy class from bundled metadata or generated source."""
        strategy_id = self._required_strategy_string(strategy_cfg, "id")
        bundled = self._bundled_class_by_id().get(strategy_id)
        if bundled is not None:
            module_name, class_name = bundled
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if not issubclass(cls, BaseStrategy):
                msg = f"{class_name} must inherit from BaseStrategy"
                raise TypeError(msg)
            return cast("type[BaseStrategy]", cls)

        class_name = self._required_strategy_string(strategy_cfg, "class_name")
        source_value = self._required_strategy_string(strategy_cfg, "source_path")
        module = self._load_module(
            self._resolve_generated_source(Path(str(source_value)))
        )
        cls = getattr(module, class_name, None)
        if cls is None:
            msg = f"Class {class_name!r} not found"
            raise ValueError(msg)
        if not issubclass(cls, BaseStrategy):
            msg = f"{class_name} must inherit from BaseStrategy"
            raise TypeError(msg)
        return cast("type[BaseStrategy]", cls)

    def _bundled_class_by_id(self) -> dict[str, tuple[str, str]]:
        """Discover bundled strategy classes from bundled config files."""
        bundled_root = Path(__file__).resolve().parent / "bundled"
        result: dict[str, tuple[str, str]] = {}
        for config_path in sorted(bundled_root.glob("*/config.yaml")):
            module_name = (
                "tradingdev.domain.strategies.bundled."
                f"{config_path.parent.name}.strategy"
            )
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            strategy = raw.get("strategy", {}) if isinstance(raw, dict) else {}
            if not isinstance(strategy, dict):
                continue
            class_name = strategy.get("class_name")
            if not isinstance(class_name, str) or not class_name:
                continue
            result[config_path.parent.name] = (module_name, class_name)
            strategy_id = strategy.get("id")
            if isinstance(strategy_id, str) and strategy_id:
                result[strategy_id] = (module_name, class_name)
        return result

    def _create_generated(
        self,
        strategy_cfg: dict[str, Any],
        engine: BaseBacktestEngine,
    ) -> BaseStrategy:
        class_name = self._required_strategy_string(strategy_cfg, "class_name")
        source_value = self._required_strategy_string(strategy_cfg, "source_path")

        source_path = self._resolve_generated_source(Path(str(source_value)))
        module = self._load_module(source_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            msg = f"Class {class_name!r} not found in {source_path}"
            raise ValueError(msg)

        instance = cls(backtest_engine=engine)
        if not isinstance(instance, BaseStrategy):
            msg = f"{class_name} must inherit from BaseStrategy"
            raise TypeError(msg)
        return instance

    def _required_strategy_string(
        self,
        strategy_cfg: dict[str, Any],
        field: str,
    ) -> str:
        value = strategy_cfg.get(field)
        if not isinstance(value, str) or not value:
            msg = f"strategy.{field} is required"
            raise ValueError(msg)
        return value

    def _resolve_generated_source(self, source_path: Path) -> Path:
        candidate = source_path
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        allowed_root = self._generated_root.resolve()
        if not resolved.is_relative_to(allowed_root):
            msg = f"Generated strategy must live under {allowed_root}"
            raise ValueError(msg)
        if not resolved.exists():
            msg = f"Strategy source not found: {resolved}"
            raise FileNotFoundError(msg)
        return resolved

    def _load_module(self, source_path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(
            f"_tradingdev_generated_{source_path.stem}",
            source_path,
        )
        if spec is None or spec.loader is None:
            msg = f"Cannot load module from {source_path}"
            raise ImportError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
