"""Provider-neutral records of visible conversation items, not research summaries."""

from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    model_validator,
)

MAX_CONVERSATION_BYTES = 32 * 1024 * 1024
type ConversationId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
type SourceId = Annotated[str, StringConstraints(min_length=1, max_length=256)]
type Outcome = Literal["running", "completed", "failed", "interrupted"]


class ConversationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, revalidate_instances="always"
    )


class ConversationItem(ConversationModel):
    source_id: SourceId
    kind: Literal["user", "assistant", "tool", "plan", "notice"]
    complete: bool = False
    text: str = ""
    # Selected public provider fields preserve tool arguments/results and phase.
    # Reasoning, credentials and hidden instructions are never accepted by adapters.
    data: dict[str, JsonValue] = {}


class ConversationTurn(ConversationModel):
    turn_id: ConversationId
    source_turn_id: SourceId | None = None
    input: Annotated[str, StringConstraints(min_length=1, max_length=65536)]
    started_at: str
    finished_at: str | None = None
    outcome: Outcome = "running"
    capture_complete: bool = False
    issues: list[str] = []
    items: list[ConversationItem] = []

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if len({item.source_id for item in self.items}) != len(self.items):
            raise ValueError("Conversation item IDs must be unique within a turn")
        if (self.outcome == "running") != (self.finished_at is None):
            raise ValueError("Only terminal turns have a finish time")
        if self.capture_complete and (
            self.outcome == "running"
            or self.source_turn_id is None
            or self.issues
            or not self.items
            or not all(item.complete for item in self.items)
        ):
            raise ValueError(
                "Complete capture requires terminal, identified, complete items"
            )
        return self


class ConversationRecord(ConversationModel):
    schema_version: Literal[1] = 1
    conversation_id: ConversationId
    version: Annotated[int, Field(strict=True, ge=1)] = 1
    provider: SourceId
    model: SourceId
    settings: dict[str, JsonValue] = {}
    source_thread_id: SourceId | None = None
    created_at: str
    updated_at: str
    turns: list[ConversationTurn] = []

    @model_validator(mode="after")
    def validate_turns(self) -> Self:
        if self.turns and self.source_thread_id is None:
            raise ValueError("Conversation turns require a bound source thread")
        if len({turn.turn_id for turn in self.turns}) != len(self.turns):
            raise ValueError("Conversation turn IDs must be unique")
        source_ids = [turn.source_turn_id for turn in self.turns if turn.source_turn_id]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Source turn IDs must be unique")
        if any(turn.outcome == "running" for turn in self.turns[:-1]):
            raise ValueError("Only the last turn can still be running")
        return self
