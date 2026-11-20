"""Public request and response models for the API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.planning.models import IntentLabel


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=8192)


class ObservationMetadata(BaseModel):
    source: str
    data_status: Literal["mock", "fresh", "stale", "unavailable"]
    captured_at: datetime


class AskResponse(BaseModel):
    request_id: str
    intent: IntentLabel
    answer: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=300)
    status: Literal[
        "classified",
        "guardrail_blocked",
        "guardrail_unavailable",
        "classification_fallback",
        "agent_unavailable",
    ]
    guardrail_status: Literal["allowed", "blocked", "unavailable"]
    guardrail_findings: list[str]
    guardrail_topics: list[str]
    classification_source: Literal["llm", "guardrail", "fallback"]
    observation: ObservationMetadata | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    llm: Literal["reachable", "unreachable", "model_unavailable"] | None = None
    guardrail_llm: Literal["reachable", "unreachable", "model_unavailable", "disabled"] | None = None
