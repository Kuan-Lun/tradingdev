"""Application service for strategy lifecycle operations."""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    now_iso,
    read_json,
    sha256_text,
    write_json,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.loader import StrategyLoader
from tradingdev.domain.strategies.schemas import (
    StrategyDiagnostic,
    StrategyMetadata,
    StrategyStatus,
    ValidationResult,
)
from tradingdev.domain.strategies.validator import (
    StrategyValidator,
    diagnostic,
    has_error,
)

_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class StrategySaveResult:
    """Result of saving a generated strategy draft."""

    success: bool
    strategy_id: str
    source_path: str
    config_path: str
    status: str
    error: str | None = None


class StrategyService:
    """Own strategy draft storage, validation, and listing."""

    def __init__(
        self,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._loader = StrategyLoader(workspace_root=self._workspace.root)
        self._validator = StrategyValidator()
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)

    def save_draft(
        self,
        strategy_id: str,
        code: str,
        yaml_config: str,
        *,
        request_summary: str = "",
    ) -> StrategySaveResult:
        """Save a generated strategy draft under workspace only."""
        if not _VALID_NAME.match(strategy_id):
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="strategy_id must be lowercase snake_case",
            )
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error=f"Python syntax error: {exc}",
            )
        try:
            parsed = yaml.safe_load(yaml_config)
        except yaml.YAMLError as exc:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error=f"YAML parse error: {exc}",
            )
        if not isinstance(parsed, dict):
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="YAML must be a mapping",
            )
        strategy_section = parsed.get("strategy", {})
        if not isinstance(strategy_section, dict):
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="YAML strategy section must be a mapping",
            )
        class_name = strategy_section.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            return StrategySaveResult(
                success=False,
                strategy_id=strategy_id,
                source_path="",
                config_path="",
                status="rejected",
                error="YAML missing required field: strategy.class_name",
            )

        source_path = self._workspace.generated_strategies / f"{strategy_id}.py"
        config_path = self._workspace.configs / f"{strategy_id}.yaml"
        metadata_path = self._metadata_path(strategy_id)
        strategy_section["id"] = strategy_id
        strategy_section["source_path"] = str(source_path)
        parsed["strategy"] = strategy_section
        normalized_yaml = yaml.safe_dump(parsed, sort_keys=False)

        source_path.write_text(code, encoding="utf-8")
        config_path.write_text(normalized_yaml, encoding="utf-8")
        created_at = now_iso()
        metadata = StrategyMetadata(
            strategy_id=strategy_id,
            class_name=class_name,
            status=StrategyStatus.DRAFT,
            created_at=created_at,
            updated_at=created_at,
            request_summary=request_summary,
            source_path=str(source_path),
            config_path=str(config_path),
            source_hash=sha256_text(code),
            config_hash=sha256_text(normalized_yaml),
        )
        self._write_metadata(metadata_path, metadata)
        return StrategySaveResult(
            success=True,
            strategy_id=strategy_id,
            source_path=str(source_path),
            config_path=str(config_path),
            status="draft",
        )

    def validate(self, strategy_id: str) -> dict[str, Any]:
        """Validate a draft strategy with static checks and a smoke dry run."""
        metadata = self._load_metadata(strategy_id)
        if metadata is None:
            return {"success": False, "error": f"Unknown strategy: {strategy_id}"}
        if metadata.status not in {StrategyStatus.DRAFT, StrategyStatus.VALIDATED}:
            return {
                "success": False,
                "strategy_id": strategy_id,
                "status": metadata.status.value,
                "error": (
                    "validate_strategy only accepts draft or validated strategies. "
                    "Use save_strategy to create a new draft before revalidating "
                    f"{metadata.status.value} strategies."
                ),
            }
        source_path = Path(metadata.source_path)
        diagnostics: list[StrategyDiagnostic] = []
        diagnostics.extend(self._validator.syntax_diagnostics(source_path))
        contract: dict[str, Any] = {"diagnostics": [], "signal_analysis": {}}
        if not self._has_error(diagnostics):
            diagnostics.extend(self._validator.static_policy_scan(source_path))
            diagnostics.extend(self._quality_gate_diagnostics(source_path))
            contract = self._run_signal_contract(metadata)
            diagnostics.extend(contract["diagnostics"])
        success = not self._has_error(diagnostics)
        metadata.validation = ValidationResult(
            checked_at=now_iso(),
            success=success,
            diagnostics=diagnostics,
            signal_analysis=contract.get("signal_analysis", {}),
        )
        metadata.status = StrategyStatus.VALIDATED if success else StrategyStatus.DRAFT
        metadata.updated_at = now_iso()
        self._write_metadata(self._metadata_path(strategy_id), metadata)
        return {
            "success": success,
            "strategy_id": strategy_id,
            "status": metadata.status.value,
            "diagnostics": [item.model_dump(mode="json") for item in diagnostics],
            "signal_analysis": contract.get("signal_analysis", {}),
        }

    def dry_run(self, strategy_id: str) -> dict[str, Any]:
        """Run a lightweight signal-generation smoke test."""
        metadata = self._load_metadata(strategy_id)
        if metadata is None:
            return {"success": False, "error": f"Unknown strategy: {strategy_id}"}
        if metadata.status != StrategyStatus.VALIDATED:
            return {
                "success": False,
                "error": "dry_run_strategy requires validated strategy status",
            }
        contract = self._run_signal_contract(metadata)
        valid = not self._has_error(contract["diagnostics"])
        metadata.status = StrategyStatus.RUNNABLE if valid else metadata.status
        metadata.dry_run = ValidationResult(
            checked_at=now_iso(),
            success=valid,
            diagnostics=contract["diagnostics"],
            signal_analysis=contract.get("signal_analysis", {}),
        )
        metadata.updated_at = now_iso()
        self._write_metadata(self._metadata_path(strategy_id), metadata)
        return {
            "success": valid,
            "strategy_id": strategy_id,
            "status": metadata.status.value,
            "diagnostics": [
                item.model_dump(mode="json") for item in contract["diagnostics"]
            ],
            "signal_analysis": contract.get("signal_analysis", {}),
        }

    def promote(self, strategy_id: str) -> dict[str, Any]:
        """Promote a runnable generated strategy."""
        metadata = self._load_metadata(strategy_id)
        if metadata is None:
            return {"success": False, "error": f"Unknown strategy: {strategy_id}"}
        if metadata.status != StrategyStatus.RUNNABLE:
            return {
                "success": False,
                "error": "Only runnable strategies can be promoted",
            }
        metadata.status = StrategyStatus.PROMOTED
        metadata.updated_at = now_iso()
        self._write_metadata(self._metadata_path(strategy_id), metadata)
        return {"success": True, "strategy_id": strategy_id, "status": "promoted"}

    def list_strategies(self) -> list[dict[str, Any]]:
        """List bundled and generated strategies."""
        items: list[dict[str, Any]] = []
        bundled_root = (
            Path(__file__).resolve().parents[1] / "domain" / "strategies" / "bundled"
        )
        for config_path in sorted(bundled_root.glob("*/config.yaml")):
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            strategy = raw.get("strategy", {}) if isinstance(raw, dict) else {}
            if isinstance(strategy, dict):
                strategy_id = str(strategy.get("id", ""))
                items.append(
                    {
                        "strategy_id": strategy_id,
                        "class_name": strategy.get("class_name"),
                        "description": strategy.get("description", ""),
                        "kind": "bundled",
                        "status": "promoted",
                        "config_path": str(config_path),
                        "metadata": {
                            "version": strategy.get("version"),
                            "source_path": strategy.get("source_path"),
                            "parameters": strategy.get("parameters", {}),
                        },
                        "data_requirements": self._data_requirements(raw),
                        "recent_runs": self._recent_runs(strategy_id),
                    }
                )
        for metadata_path in sorted(
            self._workspace.generated_strategies.glob("*.json")
        ):
            raw_metadata = read_json(metadata_path)
            metadata = (
                StrategyMetadata.model_validate(raw_metadata)
                if raw_metadata is not None
                else None
            )
            if metadata is not None:
                strategy_id = metadata.strategy_id
                config_path = Path(metadata.config_path)
                raw = (
                    yaml.safe_load(config_path.read_text(encoding="utf-8"))
                    if config_path.exists()
                    else {}
                )
                items.append(
                    {
                        "strategy_id": strategy_id,
                        "class_name": metadata.class_name,
                        "kind": "generated",
                        "status": metadata.status.value,
                        "source_path": metadata.source_path,
                        "config_path": metadata.config_path,
                        "metadata": metadata.model_dump(mode="json"),
                        "data_requirements": self._data_requirements(raw),
                        "recent_runs": self._recent_runs(strategy_id),
                    }
                )
        return items

    def get_strategy(self, strategy_id: str) -> dict[str, Any]:
        """Read bundled or generated strategy source and config."""
        metadata = self._load_metadata(strategy_id)
        if metadata is not None:
            source_path = Path(metadata.source_path)
            config_path = Path(metadata.config_path)
            return {
                "success": True,
                "strategy_id": strategy_id,
                "kind": "generated",
                "source_code": source_path.read_text(encoding="utf-8"),
                "yaml_config": config_path.read_text(encoding="utf-8"),
                "metadata": metadata.model_dump(mode="json"),
            }
        bundled = (
            Path(__file__).resolve().parents[1] / "domain" / "strategies" / "bundled"
        )
        for config_path in bundled.glob("*/config.yaml"):
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            strategy = raw.get("strategy", {}) if isinstance(raw, dict) else {}
            if isinstance(strategy, dict) and strategy.get("id") == strategy_id:
                source_path = (config_path.parent / "strategy.py").resolve()
                return {
                    "success": True,
                    "strategy_id": strategy_id,
                    "kind": "bundled",
                    "source_code": source_path.read_text(encoding="utf-8"),
                    "yaml_config": config_path.read_text(encoding="utf-8"),
                    "metadata": {
                        "status": "promoted",
                        "source_path": str(source_path),
                        "config_path": str(config_path),
                    },
                }
        return {"success": False, "error": f"Strategy not found: {strategy_id}"}

    def _metadata_path(self, strategy_id: str) -> Path:
        return self._workspace.generated_strategies / f"{strategy_id}.json"

    def _load_metadata(self, strategy_id: str) -> StrategyMetadata | None:
        raw = read_json(self._metadata_path(strategy_id))
        return StrategyMetadata.model_validate(raw) if raw is not None else None

    def _write_metadata(self, path: Path, metadata: StrategyMetadata) -> None:
        write_json(path, metadata.model_dump(mode="json"))

    def _data_requirements(self, raw_config: object) -> dict[str, Any] | None:
        if not isinstance(raw_config, dict):
            return None
        data = raw_config.get("data", {})
        if not isinstance(data, dict):
            return None
        requirements = data.get("requirements")
        return requirements if isinstance(requirements, dict) else None

    def _recent_runs(self, strategy_id: str) -> list[dict[str, Any]]:
        if not strategy_id:
            return []
        recent: list[dict[str, Any]] = []
        for run in self._store.list_runs():
            if run.get("strategy_id") != strategy_id:
                continue
            recent.append(
                {
                    "run_id": run["run_id"],
                    "job_id": run["job_id"],
                    "created_at": run["created_at"],
                    "dataset_id": run.get("dataset_id"),
                    "metrics": run.get("metrics", {}),
                }
            )
            if len(recent) >= 5:
                break
        return recent

    def _quality_gate_diagnostics(self, source_path: Path) -> list[StrategyDiagnostic]:
        diagnostics: list[StrategyDiagnostic] = []
        for command, label, timeout in (
            (["uv", "run", "ruff", "check", str(source_path)], "ruff", 30),
            (["uv", "run", "mypy", str(source_path)], "mypy", 60),
        ):
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                diagnostics.append(
                    self._diagnostic(
                        code=f"{label}_skipped",
                        phase="quality_gate",
                        message=f"{label} skipped: {exc}",
                        level="warning",
                    )
                )
                continue
            if result.returncode != 0:
                output = (result.stdout or result.stderr).strip()
                diagnostics.append(
                    self._diagnostic(
                        code=f"{label}_failed",
                        phase="quality_gate",
                        message=f"{label} failed: {output}",
                        fix=f"Fix the {label} diagnostics and save the strategy again.",
                    )
                )
        return diagnostics

    def _run_signal_contract(self, metadata: StrategyMetadata) -> dict[str, Any]:
        diagnostics: list[StrategyDiagnostic] = []
        try:
            strategy_cfg = {
                "id": metadata.strategy_id,
                "source_path": metadata.source_path,
                "class_name": metadata.class_name,
            }
            cls = self._loader.load_class(strategy_cfg)
            strategy = self._instantiate_generated(cls, metadata)
            if not isinstance(strategy, BaseStrategy):
                diagnostics.append(
                    self._diagnostic(
                        code="base_strategy_inheritance",
                        phase="contract",
                        message="strategy must inherit BaseStrategy",
                        fix="Make the generated class inherit from BaseStrategy.",
                    )
                )
                return {"diagnostics": diagnostics}
            df = self._fixture_df()
            before = df.copy(deep=True)
            result = strategy.generate_signals(df)
            if not isinstance(result, pd.DataFrame):
                diagnostics.append(
                    self._diagnostic(
                        code="signals_not_dataframe",
                        phase="contract",
                        message="generate_signals must return DataFrame",
                        fix="Return the copied DataFrame with a signal column.",
                    )
                )
                return {"diagnostics": diagnostics}
            if not df.equals(before):
                diagnostics.append(
                    self._diagnostic(
                        code="input_mutated",
                        phase="contract",
                        message="generate_signals must not mutate input",
                        fix="Start with result = df.copy() and mutate result only.",
                    )
                )
            if "signal" not in result.columns:
                diagnostics.append(
                    self._diagnostic(
                        code="missing_signal_column",
                        phase="contract",
                        message="result must include signal column",
                        fix="Add result['signal'] with values -1, 0, or 1.",
                    )
                )
                return {"diagnostics": diagnostics}
            signals = result["signal"]
            values = set(signals.dropna().unique())
            if not values.issubset({-1, 0, 1}):
                diagnostics.append(
                    self._diagnostic(
                        code="invalid_signal_values",
                        phase="contract",
                        message="signal values must be limited to -1, 0, and 1",
                        fix=(
                            "Map all generated signals to the project convention: "
                            "-1, 0, 1."
                        ),
                    )
                )
            return {
                "diagnostics": diagnostics,
                "signal_analysis": self._signal_analysis(result),
            }
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                self._diagnostic(
                    code="contract_execution_error",
                    phase="contract",
                    message=str(exc),
                    fix="Fix the class name, constructor, imports, or signal logic.",
                )
            )
            return {"diagnostics": diagnostics}

    def _instantiate_generated(
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

    def _fixture_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=80, freq="h", tz="UTC"
                ),
                "open": [100.0 + i * 0.1 for i in range(80)],
                "high": [101.0 + i * 0.1 for i in range(80)],
                "low": [99.0 + i * 0.1 for i in range(80)],
                "close": [100.5 + i * 0.1 for i in range(80)],
                "volume": [1000.0 for _ in range(80)],
            }
        )

    def _signal_analysis(self, result: pd.DataFrame) -> dict[str, Any]:
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

    def _diagnostic(
        self,
        *,
        code: str,
        phase: str,
        message: str,
        level: str = "error",
        line: int | None = None,
        fix: str | None = None,
    ) -> StrategyDiagnostic:
        return diagnostic(
            code=code,
            phase=phase,
            message=message,
            level=level,
            line=line,
            fix=fix,
        )

    def _has_error(self, diagnostics: list[StrategyDiagnostic]) -> bool:
        return has_error(diagnostics)
