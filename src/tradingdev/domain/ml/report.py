"""Project stored leaderboard rows into the shape a report is built from.

A report is assembled from a JSON payload and rendered from it; HTML is only a
projection of that payload. So a signal evaluation becomes visible in a report
by being projectable into the payload's existing scope shape, not by anyone
writing markup for it. This module is that projection and nothing else: it
reads what the leaderboard already stores and writes no new state.

Two conventions are copied deliberately from the portfolio side:

* ``values`` may hold ``None``. A candidate that was measured and produced no
  finite score stays in the payload at its own coordinate instead of vanishing
  from the grid.
* Definitions live once at block level, exactly as ``PerformanceBundle`` keeps
  them, rather than being repeated per scope.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tradingdev.domain.ml.leaderboard import LeaderboardEntry

ML_REPORT_SCHEMA_VERSION = 2
ML_SIGNAL_KIND = "ml_signal"
ML_SIGNAL_MODE = "signal"
ML_SAMPLE_PROVENANCE_NOTE = (
    "訊號來源區分後端模型、後端規則、呼叫端自備陣列與歷史未知；"
    "來源由實際評估入口記錄，不由模型名稱或訊號指紋推定。"
    "若已記錄為後端模型產生，且模型訓練資料集與評估資料集相同，"
    "則該分數為樣本內；不同不代表樣本外，兩個資料集版本仍可能涵蓋重疊期間。"
    "後端規則不涉及模型訓練；呼叫端自備或歷史未知來源的模型訓練欄位，"
    "只描述被命名的模型，不證明它產生了訊號。"
    "呼叫端自備、歷史未知來源或未保存訓練欄位時，"
    "訓練與評估的關係仍未知。來源確認本身不是樣本外證明。"
)


@dataclass(frozen=True)
class RankedSignal:
    """One placed candidate plus the entry that produced its score."""

    rank: int
    entry: LeaderboardEntry
    value: float
    tied: int


@dataclass(frozen=True)
class SkippedSignal:
    """One candidate with no usable score, and the recorded reason."""

    entry: LeaderboardEntry
    reason: str


@dataclass(frozen=True)
class SignalRanking:
    """One complete, immutable ranking bound to the entries it ranked.

    The ranked rows and the evidence are assembled together so a report cannot
    pair an old order with newer candidate content. ``market_comparison`` is
    filled by the application service, which reads the saved datasets; the
    domain projection never fetches or recomputes anything.
    """

    metric: str
    direction: str
    ranked: tuple[RankedSignal, ...]
    skipped: tuple[SkippedSignal, ...]
    caveats: tuple[str, ...]
    parameter_axes: dict[str, list[int]]
    definitions: dict[str, dict[str, Any]]
    visible_metrics: tuple[str, ...]
    hidden_metrics: tuple[str, ...]
    context: str
    families: dict[str, int]
    entries: tuple[LeaderboardEntry, ...]
    source_sha256: str
    market_comparison: Any | None = None

    @property
    def n_candidates(self) -> int:
        return len(self.entries)

    @property
    def n_ranked(self) -> int:
        return len(self.ranked)

    @property
    def n_skipped(self) -> int:
        return len(self.skipped)


def source_hash(entries: tuple[LeaderboardEntry, ...]) -> str:
    """Hash the exact selected records, not the mutable store around them."""
    source = (
        json.dumps(
            [entry.to_record() for entry in entries],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def ml_signal_scope(entry: LeaderboardEntry) -> dict[str, Any]:
    """One stored candidate as a report scope, coordinates included.

    The three tunable settings are reported in ``metadata`` rather than merged
    into ``parameters``: a fitted parameter dictionary may legitimately contain
    a key of the same name, and merging would silently overwrite one of them.
    ``parameters`` holds artifact settings for models and caller-declared
    settings for other methods; family/name preserve that distinction.
    """
    context = entry.context
    return {
        "scope_id": entry.entry_id,
        "kind": ML_SIGNAL_KIND,
        "mode": ML_SIGNAL_MODE,
        "split": "full",
        "fold_index": None,
        "trial_index": None,
        "values": {key: entry.metric(key) for key in entry.metrics},
        "parameters": dict(entry.parameters),
        "metadata": {
            "label": entry.label,
            "model_id": entry.model_id,
            "dataset_id": entry.dataset_id,
            "n_observations": entry.n_observations,
            "signal_fingerprint": entry.signal_fingerprint,
            "series_artifact_id": entry.series_artifact_id,
            "method_family": entry.family,
            "method_name": entry.method_name,
            "producer_source": entry.producer_source.value,
            "training_dataset_id": entry.training_dataset_id,
            "horizon": context.horizon,
            "window": context.window,
            "n_quantiles": context.n_quantiles,
            "symbol": context.symbol,
            "timeframe": context.timeframe,
            "start_date": context.start_date,
            "end_date": context.end_date,
            "feature_set": context.feature_set,
            "code_fingerprint": context.code_fingerprint,
            "data_fingerprint": context.data_fingerprint,
            "raw_market_version": context.raw_market_version,
        },
        # A signal evaluation records no orders and no equity curve, so there
        # is nothing to observe. Saying so is different from omitting the key.
        "observations": None,
    }


def ml_report_block(ranking: SignalRanking) -> dict[str, Any]:
    """Project one application ranking into the report payload.

    No ordering, comparability or caveat is recomputed here. The service that
    produced ``ranking`` owns those decisions, so the report cannot drift from
    the MCP answer for the same selected candidates.
    """
    block: dict[str, Any] = {
        "schema_version": ML_REPORT_SCHEMA_VERSION,
        "kind": ML_SIGNAL_KIND,
        "scopes": [ml_signal_scope(entry) for entry in ranking.entries],
        "definitions": {key: dict(value) for key, value in ranking.definitions.items()},
        # The visibility policy travels with the data so a consumer other than
        # the HTML renderer applies the same rule instead of inventing one.
        "visible_metrics": list(ranking.visible_metrics),
        "hidden_metrics": list(ranking.hidden_metrics),
        "parameter_axes": {
            key: list(value) for key, value in ranking.parameter_axes.items()
        },
        "ranking": {
            "metric": ranking.metric,
            "direction": ranking.direction,
            "ranked_scope_ids": [item.entry.entry_id for item in ranking.ranked],
            "skipped": [
                {
                    "scope_id": item.entry.entry_id,
                    "label": item.entry.label,
                    "reason": item.reason,
                    "horizon": item.entry.context.horizon,
                    "window": item.entry.context.window,
                    "n_quantiles": item.entry.context.n_quantiles,
                }
                for item in ranking.skipped
            ],
        },
        "caveats": list(ranking.caveats),
    }
    if ranking.market_comparison is not None:
        block["market_comparison"] = ranking.market_comparison
    return block
