"""Select and freeze saved ML evidence without retraining or reconstructing it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tradingdev.app.ml_service import MLService
from tradingdev.app.ranking_service import (
    MAX_SIGNAL_SELECTION,
    RankingSelectionError,
    RankingService,
)
from tradingdev.domain.ml.report import ml_report_block

if TYPE_CHECKING:
    from tradingdev.adapters.storage.filesystem import WorkspacePaths

ML_REPORT_MAX_CANDIDATES = MAX_SIGNAL_SELECTION
_RANKING_ERROR_CODES = {
    "invalid_signal_selection": "invalid_report_ml_selection",
    "signal_entry_not_found": "ml_entry_not_found",
    "signal_store_invalid": "report_ml_evidence_invalid",
}
ML_REPORT_TEMPLATE_VERSION = "9"


class MLReportError(ValueError):
    """A selection or saved-evidence failure, never permission to regenerate it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_ml_selection(
    entry_ids: list[str] | None,
    metric: str,
    allow_parameter_axes: bool,
    *,
    allow_mixed: bool = False,
    require_verified_baseline: bool = False,
) -> None:
    """Bound the explicit selection and reject options without their evidence."""
    if (
        not isinstance(metric, str)
        or not metric.strip()
        or type(allow_parameter_axes) is not bool
        or type(allow_mixed) is not bool
        or type(require_verified_baseline) is not bool
        or (
            entry_ids is None
            and (
                metric != "ic_mean"
                or allow_parameter_axes
                or allow_mixed
                or require_verified_baseline
            )
        )
    ):
        raise MLReportError(
            "invalid_report_ml_selection",
            "ML ranking options require selected saved ML entry IDs",
        )
    if entry_ids is not None and (
        not isinstance(entry_ids, list)
        or not 1 <= len(entry_ids) <= ML_REPORT_MAX_CANDIDATES
        or any(not isinstance(key, str) or not key.strip() for key in entry_ids)
        or len(set(entry_ids)) != len(entry_ids)
    ):
        raise MLReportError(
            "invalid_report_ml_selection",
            f"Provide one to {ML_REPORT_MAX_CANDIDATES} distinct saved ML entry IDs",
        )


def load_ml_report(
    workspace: WorkspacePaths,
    entry_ids: list[str],
    *,
    metric: str,
    allow_parameter_axes: bool,
    allow_mixed: bool = False,
    require_verified_baseline: bool = False,
) -> dict[str, Any]:
    """Read one complete store generation and snapshot the requested candidates.

    Re-recording an entry can change its evidence without changing its ID. The
    selected content, current definitions and ranking therefore travel inside
    the immutable report, independently of the mutable leaderboard file.
    """
    validate_ml_selection(
        entry_ids,
        metric,
        allow_parameter_axes,
        allow_mixed=allow_mixed,
        require_verified_baseline=require_verified_baseline,
    )
    try:
        service = RankingService(
            workspace=workspace, ml_service=MLService(workspace=workspace)
        )
        ranking = service.rank_signal_entries(
            entry_ids,
            metric,
            allow_parameter_axes=allow_parameter_axes,
            allow_mixed=allow_mixed,
            require_verified_baseline=require_verified_baseline,
        )
        block = ml_report_block(ranking)
    except RankingSelectionError as exc:
        code = _RANKING_ERROR_CODES.get(exc.code, "report_ml_evidence_invalid")
        raise MLReportError(code, str(exc)) from exc
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        raise MLReportError("report_ml_evidence_invalid", str(exc)) from exc
    return {
        **block,
        "entry_ids": list(entry_ids),
        "source_sha256": ranking.source_sha256,
        "template_version": ML_REPORT_TEMPLATE_VERSION,
    }
