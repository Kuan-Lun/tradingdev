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
from tradingdev.domain.strategies.base import BaseStrategy
from tradingdev.domain.strategies.loader import StrategyLoader

_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_BANNED_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "httpx",
    "ccxt",
    "shutil",
    "pathlib",
}
_ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "dataclasses",
    "datetime",
    "enum",
    "math",
    "statistics",
    "typing",
    "typing_extensions",
    "numpy",
    "pandas",
    "tradingdev",
}
_BANNED_CALLS = {
    "open",
    "eval",
    "exec",
    "__import__",
}
_BANNED_ATTR_CALLS = {
    "chmod",
    "chown",
    "mkdir",
    "remove",
    "rename",
    "replace",
    "rmdir",
    "rmtree",
    "unlink",
    "write_bytes",
    "write_text",
}


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

    def __init__(self, workspace: WorkspacePaths | None = None) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._loader = StrategyLoader(workspace_root=self._workspace.root)
        self._workspace.ensure()

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
        metadata = {
            "strategy_id": strategy_id,
            "class_name": class_name,
            "artifact_type": "generated_strategy",
            "status": "draft",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "request_summary": request_summary,
            "source_path": str(source_path),
            "config_path": str(config_path),
            "source_hash": sha256_text(code),
            "config_hash": sha256_text(normalized_yaml),
            "validation": None,
        }
        write_json(metadata_path, metadata)
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
        source_path = Path(str(metadata["source_path"]))
        diagnostics: list[dict[str, Any]] = []
        diagnostics.extend(self._syntax_diagnostics(source_path))
        contract: dict[str, Any] = {"diagnostics": [], "signal_analysis": {}}
        if not self._has_error(diagnostics):
            diagnostics.extend(self._static_policy_scan(source_path))
            diagnostics.extend(self._quality_gate_diagnostics(source_path))
            contract = self._run_signal_contract(metadata)
            diagnostics.extend(contract["diagnostics"])
        success = not self._has_error(diagnostics)
        metadata["validation"] = {
            "checked_at": now_iso(),
            "success": success,
            "diagnostics": diagnostics,
            "signal_analysis": contract.get("signal_analysis", {}),
        }
        metadata["status"] = "validated" if success else "draft"
        metadata["updated_at"] = now_iso()
        write_json(self._metadata_path(strategy_id), metadata)
        return {
            "success": success,
            "strategy_id": strategy_id,
            "status": metadata["status"],
            "diagnostics": diagnostics,
            "signal_analysis": contract.get("signal_analysis", {}),
        }

    def dry_run(self, strategy_id: str) -> dict[str, Any]:
        """Run a lightweight signal-generation smoke test."""
        metadata = self._load_metadata(strategy_id)
        if metadata is None:
            return {"success": False, "error": f"Unknown strategy: {strategy_id}"}
        if metadata.get("status") not in {"validated", "runnable", "promoted"}:
            return {
                "success": False,
                "error": "dry_run_strategy requires validated strategy status",
            }
        contract = self._run_signal_contract(metadata)
        valid = not self._has_error(contract["diagnostics"])
        metadata["status"] = "runnable" if valid else metadata["status"]
        metadata["dry_run"] = {
            "checked_at": now_iso(),
            "success": valid,
            "diagnostics": contract["diagnostics"],
            "signal_analysis": contract.get("signal_analysis", {}),
        }
        metadata["updated_at"] = now_iso()
        write_json(self._metadata_path(strategy_id), metadata)
        return {
            "success": valid,
            "strategy_id": strategy_id,
            "status": metadata["status"],
            "diagnostics": contract["diagnostics"],
            "signal_analysis": contract.get("signal_analysis", {}),
        }

    def promote(self, strategy_id: str) -> dict[str, Any]:
        """Promote a runnable generated strategy."""
        metadata = self._load_metadata(strategy_id)
        if metadata is None:
            return {"success": False, "error": f"Unknown strategy: {strategy_id}"}
        if metadata.get("status") != "runnable":
            return {
                "success": False,
                "error": "Only runnable strategies can be promoted",
            }
        metadata["status"] = "promoted"
        metadata["updated_at"] = now_iso()
        write_json(self._metadata_path(strategy_id), metadata)
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
                items.append(
                    {
                        "strategy_id": strategy.get("id"),
                        "class_name": strategy.get("class_name"),
                        "description": strategy.get("description", ""),
                        "kind": "bundled",
                        "status": "promoted",
                        "config_path": str(config_path),
                    }
                )
        for metadata_path in sorted(
            self._workspace.generated_strategies.glob("*.json")
        ):
            metadata = read_json(metadata_path)
            if metadata is not None:
                items.append(
                    {
                        "strategy_id": metadata.get("strategy_id"),
                        "class_name": metadata.get("class_name"),
                        "kind": "generated",
                        "status": metadata.get("status", "draft"),
                        "source_path": metadata.get("source_path"),
                        "config_path": metadata.get("config_path"),
                    }
                )
        return items

    def get_strategy(self, strategy_id: str) -> dict[str, Any]:
        """Read bundled or generated strategy source and config."""
        metadata = self._load_metadata(strategy_id)
        if metadata is not None:
            source_path = Path(str(metadata["source_path"]))
            config_path = Path(str(metadata["config_path"]))
            return {
                "success": True,
                "strategy_id": strategy_id,
                "kind": "generated",
                "source_code": source_path.read_text(encoding="utf-8"),
                "yaml_config": config_path.read_text(encoding="utf-8"),
                "metadata": metadata,
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

    def _load_metadata(self, strategy_id: str) -> dict[str, Any] | None:
        return read_json(self._metadata_path(strategy_id))

    def _syntax_diagnostics(self, source_path: Path) -> list[dict[str, Any]]:
        source = source_path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return [
                self._diagnostic(
                    code="syntax_error",
                    phase="syntax",
                    message=str(exc),
                    line=exc.lineno,
                    fix=(
                        "Fix Python syntax before validation can run static "
                        "policy checks."
                    ),
                )
            ]
        return []

    def _static_policy_scan(self, source_path: Path) -> list[dict[str, Any]]:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        diagnostics: list[dict[str, Any]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", maxsplit=1)[0]
                    if root in _BANNED_IMPORTS:
                        diagnostics.append(
                            self._diagnostic(
                                code="banned_import",
                                phase="static_policy",
                                message=f"banned import: {root}",
                                line=node.lineno,
                                fix=(
                                    "Remove the import; generated strategies may not "
                                    "use network, subprocess, or filesystem modules."
                                ),
                            )
                        )
                    elif root not in _ALLOWED_IMPORT_ROOTS:
                        diagnostics.append(
                            self._diagnostic(
                                code="import_not_allowed",
                                phase="static_policy",
                                message=f"import not allowed: {root}",
                                line=node.lineno,
                                fix=(
                                    "Use pandas, numpy, typing, math, datetime, or "
                                    "tradingdev strategy APIs only."
                                ),
                            )
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".", maxsplit=1)[0]
                if root in _BANNED_IMPORTS:
                    diagnostics.append(
                        self._diagnostic(
                            code="banned_import",
                            phase="static_policy",
                            message=f"banned import: {root}",
                            line=node.lineno,
                            fix=(
                                "Remove the import; generated strategies may not use "
                                "network, subprocess, or filesystem modules."
                            ),
                        )
                    )
                elif root not in _ALLOWED_IMPORT_ROOTS:
                    diagnostics.append(
                        self._diagnostic(
                            code="import_not_allowed",
                            phase="static_policy",
                            message=f"import not allowed: {root}",
                            line=node.lineno,
                            fix=(
                                "Use pandas, numpy, typing, math, datetime, or "
                                "tradingdev strategy APIs only."
                            ),
                        )
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in _BANNED_CALLS
            ):
                diagnostics.append(
                    self._diagnostic(
                        code="banned_call",
                        phase="static_policy",
                        message=f"banned call: {node.func.id}",
                        line=node.lineno,
                        fix=(
                            "Remove dynamic execution or raw file access from "
                            "strategy code."
                        ),
                    )
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _BANNED_ATTR_CALLS
            ):
                diagnostics.append(
                    self._diagnostic(
                        code="banned_attribute_call",
                        phase="static_policy",
                        message=f"banned call: {node.func.attr}",
                        line=node.lineno,
                        fix=(
                            "Generated strategies must not create, modify, or "
                            "delete files."
                        ),
                    )
                )
        return diagnostics

    def _quality_gate_diagnostics(self, source_path: Path) -> list[dict[str, Any]]:
        diagnostics: list[dict[str, Any]] = []
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

    def _run_signal_contract(self, metadata: dict[str, Any]) -> dict[str, Any]:
        diagnostics: list[dict[str, Any]] = []
        try:
            strategy_cfg = {
                "id": metadata["strategy_id"],
                "source_path": metadata["source_path"],
                "class_name": metadata["class_name"],
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
        metadata: dict[str, Any],
    ) -> BaseStrategy:
        config_path = Path(str(metadata["config_path"]))
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
    ) -> dict[str, Any]:
        diagnostic: dict[str, Any] = {
            "level": level,
            "code": code,
            "phase": phase,
            "message": message,
        }
        if line is not None:
            diagnostic["line"] = line
        if fix is not None:
            diagnostic["fix"] = fix
        return diagnostic

    def _has_error(self, diagnostics: list[dict[str, Any]]) -> bool:
        return any(item.get("level", "error") == "error" for item in diagnostics)
