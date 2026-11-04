"""Fixed confirmation models and metadata validation for captured parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from tradingdev.domain.parameter_values import effective_candidates, parameter_leaves

if TYPE_CHECKING:
    from tradingdev.domain.execution import ExecutionManifest

type NonemptyText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
type PresentationLabel = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)
]
type PresentationExplanation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
]
type PresentationPath = Annotated[str, StringConstraints(min_length=1, max_length=1024)]
MAX_PRESENTATION_PARAMETER_DESCRIPTIONS = 4096

type SectionId = Literal[
    "strategy",
    "market",
    "parameters",
    "execution",
    "validation",
    "optimization",
    "preflight",
    "limitations",
]

SECTION_TITLES: dict[SectionId, str] = {
    "strategy": "這次要測試什麼",
    "market": "市場與資料",
    "parameters": "策略設定",
    "execution": "資金、交易成本與執行設定",
    "validation": "分段驗證方式",
    "optimization": "參數搜尋方式",
    "preflight": "確認前的程式檢查",
    "limitations": "限制與待確認事項",
}


class _ConfirmationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ParameterDescription(_ConfirmationModel):
    """Model-authored wording for a parameter, never its executable value."""

    label: PresentationLabel
    description: PresentationExplanation
    unit: PresentationLabel | None = None


class ConfirmationPresentation(_ConfirmationModel):
    """Plain-language metadata bound to the exact constructor in an execution plan."""

    title: PresentationLabel
    summary: PresentationExplanation
    parameter_descriptions: dict[PresentationPath, ParameterDescription] = Field(
        max_length=MAX_PRESENTATION_PARAMETER_DESCRIPTIONS,
        description=(
            "Human labels, meanings and units for every effective constructor "
            "parameter, including defaults; never executable values. Keys are JSON "
            "pointers into constructor arguments: /fast_period for generated "
            "strategies, /config/k_period or /fit_config/... for bundled strategies. "
            "Preparation returns required_parameter_paths for missing descriptions."
        ),
    )


class ConfirmationField(_ConfirmationModel):
    """One authoritative displayed value with an optional explanation."""

    label: NonemptyText
    value_text: str
    description: str = ""
    unit: NonemptyText | None = None
    detail: bool = False
    explanation_source: Literal["backend", "model"] = "backend"


class ConfirmationSection(_ConfirmationModel):
    """A mandatory block that is shared by text and HTML output."""

    id: SectionId
    title: NonemptyText
    fields: tuple[ConfirmationField, ...] = ()
    paragraphs: tuple[str, ...] = ()
    summary_paragraphs: tuple[str, ...] = ()


class ConfirmationIdentity(_ConfirmationModel):
    """Machine identity and storage details kept outside the main explanation."""

    plan_id: str | None = None
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    details: tuple[ConfirmationField, ...] = ()


class ConfirmationDocument(_ConfirmationModel):
    """The complete built-in template, without caller-selected omissions."""

    schema_version: Literal[2] = 2
    title: NonemptyText
    summary: NonemptyText
    sections: tuple[ConfirmationSection, ...]
    identity: ConfirmationIdentity

    @model_validator(mode="after")
    def complete_template(self) -> Self:
        if tuple(section.id for section in self.sections) != tuple(SECTION_TITLES):
            raise ValueError("Confirmation requires every built-in section in order")
        for section in self.sections:
            if section.title != SECTION_TITLES[section.id]:
                raise ValueError("Confirmation section titles are fixed by the backend")
            if not section.fields and not section.paragraphs:
                raise ValueError("Confirmation sections cannot be empty")
        return self


class ConfirmationPresentationError(ValueError):
    """Missing or ambiguous metadata can be repaired without editing a revision."""

    def __init__(self, message: str, *, required_parameter_paths: list[str]) -> None:
        super().__init__(message)
        self.required_parameter_paths = required_parameter_paths


def parameter_paths(manifest: ExecutionManifest) -> list[str]:
    """Return exact JSON pointers relative to the captured constructor arguments."""
    manifest.verify()
    paths = set(parameter_leaves(manifest.strategy_execution.constructor_kwargs))
    for _group, _candidate, pointer, value in effective_candidates(manifest):
        paths.update(parameter_leaves(value, pointer))
    return sorted(paths)


def validate_presentation(
    manifest: ExecutionManifest, presentation: ConfirmationPresentation
) -> None:
    """Require unambiguous human labels for every captured parameter leaf."""
    paths = parameter_paths(manifest)
    actual = set(presentation.parameter_descriptions)
    required = set(paths)
    missing, extra = required - actual, actual - required
    if missing or extra:
        raise ConfirmationPresentationError(
            "Parameter descriptions must exactly cover the captured constructor; "
            f"missing={sorted(missing)}, extra={sorted(extra)}",
            required_parameter_paths=paths,
        )
    labels = [entry.label for entry in presentation.parameter_descriptions.values()]
    if len(labels) != len(set(labels)):
        raise ConfirmationPresentationError(
            "Parameter labels must distinguish every captured value",
            required_parameter_paths=paths,
        )
