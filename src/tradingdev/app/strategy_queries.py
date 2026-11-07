"""Read strategy identities, revisions, source, and discovery summaries."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tradingdev.adapters.storage.strategy_revisions import (
    StrategyRevisionError,
    UnsupportedStrategyRevisionError,
)
from tradingdev.app.contracts.common import DEFAULT_PAGE_LIMIT, Page
from tradingdev.app.contracts.strategy import (
    BundledStrategySummary,
    GeneratedStrategySummary,
    LegacyStrategySummary,
    StrategyKind,
    StrategySummary,
)
from tradingdev.app.pagination import paginate
from tradingdev.app.strategy_errors import strategy_not_found, strategy_revision_error
from tradingdev.domain.strategies.capabilities import StrategyCapabilities
from tradingdev.domain.strategies.schemas import StrategySpec, StrategyStatus

if TYPE_CHECKING:
    from tradingdev.adapters.storage.sqlite import SQLiteStore
    from tradingdev.adapters.storage.strategy_revisions import StrategyRevisionStore
    from tradingdev.domain.strategies.catalog import BundledStrategyCatalog


class StrategyQueries:
    """Project the catalog and revision store without changing lifecycle state."""

    def __init__(
        self,
        catalog: BundledStrategyCatalog,
        revisions: StrategyRevisionStore,
        store: SQLiteStore,
    ) -> None:
        self._catalog = catalog
        self._revisions = revisions
        self._store = store

    def load(
        self, strategy_id: str, revision_id: str | None = None
    ) -> StrategySpec | None:
        """Load a bundled or generated strategy spec."""
        entry = self._catalog.get(strategy_id) if revision_id is None else None
        metadata = (
            None
            if entry is not None
            else self._revisions.load(strategy_id, revision_id)
        )
        if metadata is not None:
            return StrategySpec(
                strategy_id=metadata.strategy_id,
                revision_id=metadata.revision_id,
                class_name=metadata.class_name,
                source_path=metadata.source_path,
                config_path=metadata.config_path,
                status=metadata.status,
                kind="generated",
                metadata=metadata,
            )

        if revision_id is not None:
            return None
        if entry is None:
            return None
        return StrategySpec(
            strategy_id=strategy_id,
            class_name=entry.class_name,
            source_path=str(entry.module_source_path),
            config_path=str(entry.config_path),
            status=StrategyStatus.PROMOTED,
            kind="bundled",
            execution_capabilities=entry.execution_capabilities,
            metadata={
                "version": entry.strategy_section.get("version"),
                "declared_source_path": entry.declared_source_path,
            },
        )

    def list_strategies(self) -> list[dict[str, Any]]:
        """List bundled, revisioned, and legacy strategies requiring resaving."""
        items: list[dict[str, Any]] = []
        for entry in self._catalog.entries():
            strategy = entry.strategy_section
            items.append(
                {
                    "strategy_id": entry.strategy_id,
                    "class_name": entry.class_name,
                    "description": strategy.get("description", ""),
                    "kind": "bundled",
                    "execution_capabilities": entry.execution_capabilities.model_dump(
                        mode="json"
                    ),
                    "revision_id": None,
                    "status": "promoted",
                    "config_path": str(entry.config_path),
                    "metadata": {
                        "revision_id": None,
                        "version": strategy.get("version"),
                        "source_path": str(entry.module_source_path),
                        "parameters": strategy.get("parameters", {}),
                    },
                    "data_requirements": self._data_requirements(entry.raw_config),
                    "recent_runs": self._recent_runs(entry.strategy_id),
                }
            )
        for metadata in self._revisions.list_current():
            raw = yaml.safe_load(Path(metadata.config_path).read_text(encoding="utf-8"))
            items.append(
                {
                    "strategy_id": metadata.strategy_id,
                    "revision_id": metadata.revision_id,
                    "class_name": metadata.class_name,
                    "kind": "generated",
                    "execution_capabilities": StrategyCapabilities().model_dump(
                        mode="json"
                    ),
                    "status": metadata.status.value,
                    "source_path": metadata.source_path,
                    "config_path": metadata.config_path,
                    "metadata": metadata.model_dump(mode="json"),
                    "data_requirements": self._data_requirements(raw),
                    "recent_runs": self._recent_runs(metadata.strategy_id),
                }
            )
        items.extend(
            self._legacy_summary(strategy_id)
            for strategy_id in self._revisions.list_legacy_ids()
        )
        return items

    def _legacy_summary(self, strategy_id: str) -> dict[str, Any]:
        reserved = self._catalog.get(strategy_id) is not None
        read_step = "get_strategy(legacy=true)" if reserved else "get_strategy"
        save_step = (
            "save_strategy using a different, non-reserved strategy_id"
            if reserved
            else "save_strategy"
        )
        return {
            "strategy_id": strategy_id,
            "revision_id": None,
            "kind": "legacy",
            "execution_capabilities": StrategyCapabilities().model_dump(mode="json"),
            "status": "revision_required",
            "code": "strategy_revision_required",
            "message": (
                f"Read source/config with {read_step}, then explicitly {save_step} "
                "as a new draft and validate/dry-run its returned revision_id. "
                "Legacy status and evidence do not authorize execution."
            ),
        }

    def list_strategies_page(
        self,
        *,
        kind: StrategyKind | None = None,
        strategy_id: str | None = None,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Page[StrategySummary]:
        """Discover identities without expanding configuration or run history."""
        entries: list[StrategySummary] = []
        if kind in (None, "bundled"):
            entries.extend(
                BundledStrategySummary(
                    strategy_id=entry.strategy_id,
                    revision_id=None,
                    class_name=entry.class_name,
                    description=entry.strategy_section.get("description", ""),
                    kind="bundled",
                    execution_capabilities=entry.execution_capabilities,
                    status="promoted",
                )
                for entry in self._catalog.entries()
                if strategy_id is None or entry.strategy_id == strategy_id
            )
        if kind in (None, "generated"):
            entries.extend(
                GeneratedStrategySummary(
                    strategy_id=metadata.strategy_id,
                    revision_id=metadata.revision_id,
                    class_name=metadata.class_name,
                    kind="generated",
                    execution_capabilities=StrategyCapabilities(),
                    status=metadata.status,
                )
                for metadata in self._revisions.list_current()
                if strategy_id is None or metadata.strategy_id == strategy_id
            )
        if kind in (None, "legacy"):
            entries.extend(
                LegacyStrategySummary.model_validate(self._legacy_summary(identity))
                for identity in self._revisions.list_legacy_ids()
                if strategy_id is None or identity == strategy_id
            )
        entries.sort(
            key=lambda item: (item.strategy_id, item.kind, item.revision_id or "")
        )
        return paginate(entries, project=lambda item: item, offset=offset, limit=limit)

    def get_strategy(
        self,
        strategy_id: str,
        revision_id: str | None = None,
        *,
        legacy: bool = False,
    ) -> dict[str, Any]:
        """Read bundled or generated strategy source and config."""
        if legacy:
            if revision_id is not None:
                return strategy_revision_error(
                    StrategyRevisionError("Legacy source has no revision_id")
                )
            if strategy_id not in self._revisions.list_legacy_ids():
                return strategy_not_found(strategy_id, None)
            return self._get_legacy_strategy(strategy_id)
        try:
            metadata = (
                None
                if revision_id is None and self._catalog.get(strategy_id) is not None
                else self._revisions.load(strategy_id, revision_id)
            )
        except UnsupportedStrategyRevisionError:
            return self._get_legacy_strategy(strategy_id)
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)
        if metadata is not None:
            source_path = Path(metadata.source_path)
            config_path = Path(metadata.config_path)
            return {
                "success": True,
                "strategy_id": strategy_id,
                "kind": "generated",
                "execution_capabilities": StrategyCapabilities().model_dump(
                    mode="json"
                ),
                "revision_id": metadata.revision_id,
                "source_code": source_path.read_text(encoding="utf-8"),
                "yaml_config": config_path.read_text(encoding="utf-8"),
                "metadata": metadata.model_dump(mode="json"),
            }
        if revision_id is not None:
            return strategy_not_found(strategy_id, revision_id)
        entry = self._catalog.get(strategy_id)
        if entry is not None:
            return {
                "success": True,
                "strategy_id": strategy_id,
                "kind": "bundled",
                "execution_capabilities": entry.execution_capabilities.model_dump(
                    mode="json"
                ),
                "revision_id": None,
                "source_code": entry.module_source_path.read_text(encoding="utf-8"),
                "yaml_config": entry.config_path.read_text(encoding="utf-8"),
                "metadata": {
                    "status": "promoted",
                    "revision_id": None,
                    "source_path": str(entry.module_source_path),
                    "config_path": str(entry.config_path),
                },
            }
        return {
            "success": False,
            "error": f"Strategy not found: {strategy_id}",
            "code": "strategy_not_found",
        }

    def _get_legacy_strategy(self, strategy_id: str) -> dict[str, Any]:
        try:
            source, config = self._revisions.read_legacy_source(strategy_id)
        except StrategyRevisionError as exc:
            return strategy_revision_error(exc)
        return {
            **self._legacy_summary(strategy_id),
            "success": True,
            "source_code": source,
            "yaml_config": config,
        }

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
                    "revision_id": run.get("revision_id"),
                    "created_at": run["created_at"],
                    "dataset_id": run.get("dataset_id"),
                    "metrics": run.get("metrics", {}),
                }
            )
            if len(recent) >= 5:
                break
        return recent
