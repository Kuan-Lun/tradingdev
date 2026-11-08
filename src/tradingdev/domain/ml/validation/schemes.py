"""Named purged validation schemes shared by discovery and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tradingdev.domain.ml.validation.splits import (
    leakage_audit,
    purged_kfold_splits,
    purged_walk_forward_splits,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradingdev.domain.ml.validation.splits import Split


class SplitBuilder(Protocol):
    """Uniform fold-count and purging contract for a registered builder."""

    def __call__(
        self,
        n_bars: int,
        *,
        horizon: int,
        n_folds: int = 5,
        embargo: int = 0,
        min_train_bars: int = 1,
    ) -> list[Split]: ...


@dataclass(frozen=True)
class ValidationScheme:
    """Named folds whose builder always accepts the ``n_folds`` keyword."""

    name: str
    description: str
    build: SplitBuilder


def _walk_forward_splits(
    n_bars: int,
    *,
    horizon: int,
    n_folds: int = 5,
    embargo: int = 0,
    min_train_bars: int = 1,
) -> list[Split]:
    return purged_walk_forward_splits(
        n_bars,
        horizon=horizon,
        n_splits=n_folds,
        embargo=embargo,
        min_train_bars=min_train_bars,
    )


_SCHEMES = (
    ValidationScheme(
        name="purged_walk_forward",
        description=(
            "Anchored walk-forward with purged/embargoed training bars; "
            "block 0 always trains, each later block is tested once."
        ),
        build=_walk_forward_splits,
    ),
    ValidationScheme(
        name="purged_kfold",
        description=(
            "Every bar falls in exactly one test window; training spans both "
            "sides with the label horizon purged and an embargo gap. "
            "Retrospective cross-validation, not a past-only deployment simulation."
        ),
        build=purged_kfold_splits,
    ),
)

VALIDATION_SCHEMES: dict[str, ValidationScheme] = {
    scheme.name: scheme for scheme in _SCHEMES
}


def validation_scheme_names() -> tuple[str, ...]:
    """Names of the schemes that can be asked for, in a stable order."""
    return tuple(scheme.name for scheme in _SCHEMES)


def get_validation_scheme(name: str) -> ValidationScheme:
    """Resolve a scheme; reject unknown names with the available alternatives."""
    try:
        return VALIDATION_SCHEMES[name]
    except KeyError:
        raise ValueError(
            f"unknown validation scheme: {name}; "
            f"choose one of {', '.join(validation_scheme_names())}"
        ) from None


def build_validation_splits(
    name: str,
    *,
    n_bars: int,
    horizon: int,
    n_folds: int = 5,
    embargo: int = 0,
    min_train_bars: int = 1,
) -> list[Split]:
    """Build named folds over ``n_bars``, purging by label horizon.

    ``embargo`` adds excluded bars; ``min_train_bars`` bounds retained train
    size. Unknown schemes and unusable fold geometry raise ValueError.
    """
    scheme = get_validation_scheme(name)
    return scheme.build(
        n_bars,
        horizon=horizon,
        embargo=embargo,
        min_train_bars=min_train_bars,
        n_folds=n_folds,
    )


def audit_validation_splits(
    splits: Sequence[Split],
    *,
    horizon: int,
) -> tuple[str, ...]:
    """Audit constructed folds for label overlap before fitting any model."""
    return tuple(leakage_audit(splits, horizon=horizon))
