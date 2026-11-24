"""Explicit representative creation, revision and historical lookup."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mcp_types import ToolAnnotations
from pydantic import TypeAdapter

from tradingdev.app.contracts.common import ErrorResponse, PageLimit, PageOffset
from tradingdev.app.contracts.representatives import (
    RepresentativeList,
    RepresentativeRead,
    RepresentativeSaved,
)
from tradingdev.domain.representatives import (  # noqa: TC001 -- MCP runtime schema
    RepresentativeId,
    RepresentativeVersion,
    ResearchGroup,
    SavedRunId,
    SavedScope,
    SelectionReason,
)

if TYPE_CHECKING:
    from mcp.server import MCPServer

    from tradingdev.app.representative_service import RepresentativeService


def register(mcp: MCPServer, service: RepresentativeService) -> None:
    write = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    @mcp.tool(annotations=write)
    def create_strategy_representative(
        group_id: ResearchGroup,
        run_id: SavedRunId,
        scope: SavedScope,
        reason: SelectionReason = "",
    ) -> RepresentativeSaved | ErrorResponse:
        """Save a user's explicit run/scope choice for one research group/strategy.

        New only; existing choices require read then revise. Never infer a choice
        from latest/highest score. See get_ml_contract guidance.representatives.
        """
        return TypeAdapter(RepresentativeSaved | ErrorResponse).validate_python(
            service.create(group_id, run_id, scope, reason)
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=False,
            open_world_hint=False,
        )
    )
    def revise_strategy_representative(
        representative_id: RepresentativeId,
        expected_version: RepresentativeVersion,
        run_id: SavedRunId,
        scope: SavedScope,
        reason: SelectionReason = "",
    ) -> RepresentativeSaved | ErrorResponse:
        """Explicitly reselect a representative with its expected current version.

        Preserves history; stale versions conflict. Do not retry with a newer
        version or overwrite after conflict without a new explicit user choice.
        """
        return TypeAdapter(RepresentativeSaved | ErrorResponse).validate_python(
            service.revise(representative_id, expected_version, run_id, scope, reason)
        )

    @mcp.tool(
        annotations=ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        )
    )
    def get_strategy_representatives(
        representative_id: RepresentativeId | None = None,
        version: RepresentativeVersion | None = None,
        group_id: ResearchGroup | None = None,
        run_id: SavedRunId | None = None,
        offset: PageOffset = 0,
        limit: PageLimit = 20,
    ) -> RepresentativeRead | RepresentativeList | ErrorResponse:
        """Read current/explicit historical version by ID, or page current choices.

        run_id lists all historical references for retention; follow next_offset.
        Missing sources keep snapshots but cannot rank. Never reselect on reads.
        """
        return TypeAdapter(
            RepresentativeRead | RepresentativeList | ErrorResponse
        ).validate_json(
            json.dumps(
                service.get(
                    representative_id,
                    version,
                    group_id=group_id,
                    run_id=run_id,
                    offset=offset,
                    limit=limit,
                ),
                allow_nan=False,
            )
        )
