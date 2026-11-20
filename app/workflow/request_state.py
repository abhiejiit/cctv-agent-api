"""Mutable request-scoped workflow state, independent of any agent framework."""

from pydantic import BaseModel, ConfigDict, Field

from app.auth.principal import Principal
from app.planning.models import IntentLabel
from app.services.conversation_history_service import ConversationTurn


class RequestState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    principal: Principal
    user_message: str
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    conversation_history_status: str = "pending"
    normalized_message: str = ""
    guardrail_status: str = "pending"
    guardrail_findings: list[str] = Field(default_factory=list)
    guardrail_topics: list[str] = Field(default_factory=list)
    intent: IntentLabel | None = None
    classification_reason: str | None = None
    classification_source: str | None = None
    agent_response: str | None = None
    observation_metadata: dict[str, str] | None = None
    workflow_status: str = "received"
