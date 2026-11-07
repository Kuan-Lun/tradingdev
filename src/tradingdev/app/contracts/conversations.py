"""Conversation discovery and immutable reading artifact identities."""

from tradingdev.app.contracts.common import ContractModel
from tradingdev.domain.conversations import ConversationId, SourceId


class ConversationSummary(ContractModel):
    conversation_id: ConversationId
    version: int
    provider: SourceId
    model: SourceId
    turns: int
    updated_at: str


class ConversationExport(ContractModel):
    conversation_id: str
    version: int
    export_id: str
    artifact_id: str
    html_path: str
    json_path: str
    sha256: str
