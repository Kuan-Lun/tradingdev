"""Application service for strategy lifecycle operations."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import yaml

from tradingdev.adapters.execution.strategy_quality import run_quality_gates
from tradingdev.adapters.storage.filesystem import (
    WorkspacePaths,
    now_iso,
)
from tradingdev.adapters.storage.sqlite import SQLiteStore, get_sqlite_store
from tradingdev.adapters.storage.strategy_revisions import (
    StrategyNamespaceOccupiedError,
    StrategyRevisionConflictError,
    StrategyRevisionError,
    StrategyRevisionStore,
)
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.quality_policy import quality_check_config
from tradingdev.app.research import SelectedResearchRevision, build_research_config
from tradingdev.app.strategy_contract_service import StrategyContractService
from tradingdev.app.strategy_errors import strategy_not_found, strategy_revision_error
from tradingdev.app.strategy_queries import StrategyQueries
from tradingdev.domain.research import ResearchSpec  # noqa: TC001
from tradingdev.domain.strategies.catalog import BundledStrategyCatalog
from tradingdev.domain.strategies.contract import (
    DRY_RUN_FIXTURE_ROWS,
    VALIDATE_FIXTURE_ROWS,
)
from tradingdev.domain.strategies.schemas import (
    StrategyDiagnostic,
    StrategyMetadata,
    StrategySpec,
    StrategyStatus,
    ValidationResult,
)
from tradingdev.domain.strategies.validator import (
    StrategyValidator,
    has_error,
)

if TYPE_CHECKING:
    from tradingdev.app.contracts.strategy import StrategyKind, StrategySummary

_VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

_EXECUTABLE_STATUSES = {StrategyStatus.RUNNABLE, StrategyStatus.PROMOTED}


class StrategyNotExecutableError(RuntimeError):
    """Raised when a strategy has not reached an executable lifecycle status."""


@dataclass(frozen=True)
class StrategySaveResult:
    """Result of saving a generated strategy draft."""

    success: bool
    strategy_id: str
    source_path: str
    config_path: str
    status: str
    revision_id: str | None = None
    error: str | None = None
    code: str | None = None


class StrategyService:
    """Own strategy draft storage, validation, and listing."""

    def __init__(
        self,
        workspace: WorkspacePaths | None = None,
        store: SQLiteStore | None = None,
        *,
        contract_timeout_seconds: float = 60,
    ) -> None:
        self._workspace = workspace or WorkspacePaths()
        self._catalog = BundledStrategyCatalog()
        self._validator = StrategyValidator()
        self._contract_checker = StrategyContractService(
            self._workspace, timeout_seconds=contract_timeout_seconds
        )
        self._workspace.ensure()
        self._store = store or get_sqlite_store(self._workspace)
        self._revisions = StrategyRevisionStore(self._workspace)
        self._queries = StrategyQueries(self._catalog, self._revisions, self._store)

    def save_draft(
        self,
        code: str,
        yaml_config: str,
        research: ResearchSpec,
        *,
        strategy_id: str | None = None,
        expected_revision_id: str | None = None,
        display_name: str | None = None,
    ) -> StrategySaveResult:
        """Create a strategy or explicitly revise the selected current revision."""
        candidate = self._new_strategy_id() if strategy_id is None else strategy_id
        if strategy_id is None and expected_revision_id is not None:
            return self._save_error(
                candidate,
                "invalid_revision_request",
                "expected_revision_id requires an explicit strategy_id",
            )
        if not _VALID_NAME.fullmatch(candidate):
            return self._save_error(
                candidate,
                "invalid_strategy_id",
                "strategy_id must be lowercase snake_case",
            )
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return self._save_error(
                candidate, "syntax_error", f"Python syntax error: {exc}"
            )
        try:
            parsed = yaml.safe_load(yaml_config)
        except yaml.YAMLError as exc:
            return self._save_error(
                candidate, "invalid_yaml", f"YAML parse error: {exc}"
            )
        if not isinstance(parsed, dict):
            return self._save_error(
                candidate, "invalid_strategy_config", "YAML must be a mapping"
            )
        section = parsed.get("strategy")
        if not isinstance(section, dict):
            return self._save_error(
                candidate,
                "invalid_strategy_config",
                "YAML strategy section must be a mapping",
            )
        class_name = section.get("class_name")
        if not isinstance(class_name, str) or not class_name:
            return self._save_error(
                candidate,
                "invalid_strategy_config",
                "YAML missing required field: strategy.class_name",
            )
        if display_name is not None and not display_name.strip():
            return self._save_error(
                candidate, "invalid_strategy_config", "display_name must not be empty"
            )

        for attempt in range(8 if strategy_id is None else 1):
            if attempt:
                candidate = self._new_strategy_id()
            try:
                with self._revisions.lifecycle_lock(candidate):
                    if self._catalog.get(candidate) is not None:
                        if strategy_id is None:
                            continue
                        return self._save_error(
                            candidate,
                            "reserved_strategy_id",
                            "Generated strategies cannot replace a bundled strategy ID",
                        )
                    previous = self._revisions.check_save_target(
                        candidate, expected_revision_id
                    )
                    selected = None
                    if previous is not None:
                        selected = SelectedResearchRevision(
                            strategy_id=candidate,
                            revision_id=previous.revision_id,
                            config_hash=previous.config_hash,
                            config=yaml.safe_load(
                                Path(previous.config_path).read_text(encoding="utf-8")
                            ),
                        )
                    try:
                        config = build_research_config(
                            parsed, research, selected_revision=selected
                        )
                    except ValueError as exc:
                        return self._save_error(
                            candidate, "invalid_research_spec", str(exc)
                        )
                    outline = config["research"]["strategy_outline"]
                    captured_name = config["strategy"].get("display_name")
                    if captured_name is not None and (
                        not isinstance(captured_name, str) or not captured_name.strip()
                    ):
                        return self._save_error(
                            candidate,
                            "invalid_strategy_config",
                            "YAML strategy.display_name must be nonempty text",
                        )
                    if (
                        display_name is not None
                        and captured_name is not None
                        and captured_name.strip() != display_name.strip()
                    ):
                        return self._save_error(
                            candidate,
                            "invalid_strategy_config",
                            "YAML strategy.display_name conflicts with display_name",
                        )
                    name = display_name if display_name is not None else captured_name
                    config["strategy"]["display_name"] = (
                        name.strip()
                        if name is not None
                        else outline.splitlines()[0][:80]
                    )
                    metadata = self._revisions.create(
                        candidate,
                        code,
                        config,
                        request_summary=outline,
                        expected_revision_id=expected_revision_id,
                    )
            except StrategyNamespaceOccupiedError as exc:
                if strategy_id is None:
                    continue
                return self._save_error(candidate, "strategy_id_exists", str(exc))
            except StrategyRevisionConflictError as exc:
                return self._save_error(
                    candidate, "strategy_revision_conflict", str(exc)
                )
            except StrategyRevisionError as exc:
                return self._save_error(
                    candidate, "strategy_revision_invalid", str(exc)
                )
            except OSError as exc:
                return self._save_error(candidate, "strategy_save_failed", str(exc))
            return StrategySaveResult(
                success=True,
                strategy_id=candidate,
                revision_id=metadata.revision_id,
                source_path=metadata.source_path,
                config_path=metadata.config_path,
                status="draft",
            )
        return self._save_error(
            candidate,
            "strategy_id_generation_failed",
            "Could not allocate an unused strategy ID; retry the save",
        )

    @staticmethod
    def _new_strategy_id() -> str:
        return "strategy_" + uuid4().hex

    @staticmethod
    def _save_error(strategy_id: str, code: str, message: str) -> StrategySaveResult:
        return StrategySaveResult(
            success=False,
            strategy_id=strategy_id,
            source_path="",
            config_path="",
            status="rejected",
            code=code,
            error=message,
        )

    def load(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec | None:
        """Load a bundled or generated strategy spec."""
        return self._queries.load(strategy_id, revision_id)

    def resolve_executable(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec:
        """Return the spec for a strategy allowed to execute backtests.

        Raises:
            StrategyNotExecutableError: If the strategy is unknown or has not
                reached runnable or promoted status.
        """
        try:
            spec = self.load(strategy_id, revision_id)
        except StrategyRevisionError as exc:
            raise StrategyNotExecutableError(str(exc)) from exc
        if spec is None:
            msg = f"Strategy not found: {strategy_id}, revision={revision_id}"
            raise StrategyNotExecutableError(msg)
        if spec.status not in _EXECUTABLE_STATUSES:
            msg = (
                "Strategy must be runnable or promoted before execution. "
                f"Current status: {spec.status.value}"
            )
            raise StrategyNotExecutableError(msg)
        if isinstance(spec.metadata, StrategyMetadata):
            checks = (spec.metadata.validation, spec.metadata.dry_run)
            if any(
                check is None
                or not check.success
                or check.has_error
                or check.revision_id != spec.revision_id
                for check in checks
            ):
                msg = "Strategy revision requires successful validation and dry-run"
                raise StrategyNotExecutableError(msg)
        return spec

    def record_validation_status(
        self,
        strategy_id: str,
        result: ValidationResult | dict[str, Any],
    ) -> dict[str, Any]:
        """Record external evidence against the revision it actually checked."""
        try:
            with self._revisions.lifecycle_lock(strategy_id):
                return self._record_validation_status(strategy_id, result)
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)

    def _record_validation_status(
        self,
        strategy_id: str,
        result: ValidationResult | dict[str, Any],
    ) -> dict[str, Any]:
        validation = (
            result
            if isinstance(result, ValidationResult)
            else ValidationResult.model_validate(result)
        )
        try:
            metadata = self._revisions.load(strategy_id, validation.revision_id)
            if metadata is None:
                return strategy_not_found(strategy_id, validation.revision_id)
            if metadata.status not in {StrategyStatus.DRAFT, StrategyStatus.VALIDATED}:
                return self._state_error(
                    metadata,
                    "record_validation_status only accepts draft or validated "
                    "strategies.",
                )
            expected_metadata = metadata.model_copy(deep=True)
            if validation.has_error:
                validation = validation.model_copy(update={"success": False})
            metadata.validation = validation
            metadata.dry_run = None
            metadata.status = (
                StrategyStatus.VALIDATED if validation.success else StrategyStatus.DRAFT
            )
            metadata.updated_at = now_iso()
            self._revisions.update(metadata, expected_metadata=expected_metadata)
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)
        return self._check_response(metadata, validation)

    def validate(
        self, strategy_id: str, revision_id: str | None = None
    ) -> dict[str, Any]:
        """Validate the selected immutable revision with static and runtime checks."""
        return self._check_revision(strategy_id, revision_id, dry_run=False)

    def dry_run(
        self, strategy_id: str, revision_id: str | None = None
    ) -> dict[str, Any]:
        """Run the longer fixture against the revision's successful validation."""
        return self._check_revision(strategy_id, revision_id, dry_run=True)

    def _check_revision(
        self, strategy_id: str, revision_id: str | None, *, dry_run: bool
    ) -> dict[str, Any]:
        try:
            with self._revisions.lifecycle_lock(strategy_id):
                return self._check_locked_revision(
                    strategy_id, revision_id, dry_run=dry_run
                )
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)

    def _check_locked_revision(
        self, strategy_id: str, revision_id: str | None, *, dry_run: bool
    ) -> dict[str, Any]:
        try:
            metadata = self._revisions.load(strategy_id, revision_id)
            if metadata is None:
                return strategy_not_found(strategy_id, revision_id)
            if dry_run:
                if metadata.status != StrategyStatus.VALIDATED:
                    return self._state_error(
                        metadata, "dry_run_strategy requires validated strategy status"
                    )
                evidence = metadata.validation
                if (
                    evidence is None
                    or not evidence.success
                    or evidence.has_error
                    or evidence.revision_id != metadata.revision_id
                ):
                    return self._state_error(
                        metadata, "This revision requires successful validation first"
                    )
            elif metadata.status not in {
                StrategyStatus.DRAFT,
                StrategyStatus.VALIDATED,
            }:
                return self._state_error(
                    metadata,
                    "validate_strategy only accepts draft or validated strategies. "
                    "Use save_strategy to create a new draft before revalidating "
                    f"{metadata.status.value} strategies.",
                )
            expected_metadata = metadata.model_copy(deep=True)
            self._revisions.verify(metadata)
            source_path = Path(metadata.source_path)
            diagnostics: list[StrategyDiagnostic] = []
            signal_analysis: dict[str, Any] = {}
            effective_config: dict[str, Any] | None = None
            if not dry_run:
                diagnostics.extend(self._validator.syntax_diagnostics(source_path))
                if not has_error(diagnostics):
                    diagnostics.extend(self._validator.static_policy_scan(source_path))
                if not has_error(diagnostics):
                    diagnostics.extend(self._quality_gate_diagnostics(source_path))
            if not has_error(diagnostics):
                contract = self._contract_checker.check(
                    metadata,
                    fixture_rows=DRY_RUN_FIXTURE_ROWS
                    if dry_run
                    else VALIDATE_FIXTURE_ROWS,
                )
                diagnostics.extend(contract["diagnostics"])
                signal_analysis = contract.get("signal_analysis", {})
                effective_config = contract.get("effective_config")
            # A new current revision may have been saved while the check ran.
            # Only this revision is updated, and modified snapshot bytes are rejected.
            self._revisions.verify(metadata)
            result = ValidationResult(
                revision_id=metadata.revision_id,
                checked_at=now_iso(),
                success=not has_error(diagnostics),
                diagnostics=diagnostics,
                signal_analysis=signal_analysis,
                effective_config=effective_config,
            )
            if dry_run:
                metadata.dry_run = result
                if result.success:
                    metadata.status = StrategyStatus.RUNNABLE
            else:
                metadata.validation = result
                metadata.dry_run = None
                metadata.status = (
                    StrategyStatus.VALIDATED if result.success else StrategyStatus.DRAFT
                )
            metadata.updated_at = now_iso()
            self._revisions.update(metadata, expected_metadata=expected_metadata)
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)
        return self._check_response(metadata, result)

    def promote(
        self, strategy_id: str, revision_id: str | None = None
    ) -> dict[str, Any]:
        """Promote a runnable revision without changing the current revision pointer."""
        try:
            metadata = self._revisions.load(strategy_id, revision_id)
            if metadata is None:
                return strategy_not_found(strategy_id, revision_id)
            if metadata.status != StrategyStatus.RUNNABLE:
                return self._state_error(
                    metadata, "Only runnable strategies can be promoted"
                )
            self.resolve_executable(strategy_id, metadata.revision_id)
            expected_metadata = metadata.model_copy(deep=True)
            metadata.status = StrategyStatus.PROMOTED
            metadata.updated_at = now_iso()
            self._revisions.update(metadata, expected_metadata=expected_metadata)
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)
        except StrategyNotExecutableError as exc:
            return {
                "success": False,
                "error": str(exc),
                "code": "strategy_not_executable",
            }
        return {
            "success": True,
            "strategy_id": strategy_id,
            "revision_id": metadata.revision_id,
            "status": "promoted",
        }

    @staticmethod
    def _check_response(
        metadata: StrategyMetadata, result: ValidationResult
    ) -> dict[str, Any]:
        return {
            "success": result.success,
            "strategy_id": metadata.strategy_id,
            "revision_id": metadata.revision_id,
            "status": metadata.status.value,
            "diagnostics": [
                item.model_dump(mode="json") for item in result.diagnostics
            ],
            "signal_analysis": result.signal_analysis,
            "effective_config": result.effective_config,
        }

    @staticmethod
    def _state_error(metadata: StrategyMetadata, message: str) -> dict[str, Any]:
        return {
            "success": False,
            "strategy_id": metadata.strategy_id,
            "revision_id": metadata.revision_id,
            "status": metadata.status.value,
            "error": message,
            "code": "invalid_strategy_status",
        }

    def list_strategies(self) -> list[dict[str, Any]]:
        """List bundled, revisioned, and legacy strategies requiring resaving."""
        return self._queries.list_strategies()

    def list_strategies_page(
        self,
        *,
        kind: StrategyKind | None = None,
        strategy_id: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[StrategySummary]:
        """Discover identities without expanding configuration or run history."""
        return self._queries.list_strategies_page(
            kind=kind, strategy_id=strategy_id, offset=offset, limit=limit
        )

    def get_strategy(
        self,
        strategy_id: str,
        revision_id: str | None = None,
        *,
        legacy: bool = False,
    ) -> dict[str, Any]:
        """Read bundled or generated strategy source and config."""
        return self._queries.get_strategy(strategy_id, revision_id, legacy=legacy)

    def _quality_gate_diagnostics(self, source_path: Path) -> list[StrategyDiagnostic]:
        with quality_check_config() as config:
            return run_quality_gates(source_path, config)
