"""Public request and response models for the API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.planning.models import IntentLabel


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    request_id: str
    intent: IntentLabel
    reason: str = Field(min_length=1, max_length=300)
    status: Literal[
        "classified",
        "guardrail_blocked",
        "guardrail_unavailable",
        "classification_fallback",
    ]
    guardrail_status: Literal["allowed", "blocked", "unavailable"]
    guardrail_findings: list[str]
    guardrail_topics: list[str]
    classification_source: Literal["llm", "guardrail", "fallback"]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    llm: Literal["reachable", "unreachable"] | None = None
