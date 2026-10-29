"""Types shared by intent classification and the later orchestration flow."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentLabel(str, Enum):
    FRONT_ENTRANCE = "front_entrance"
    PARKING = "parking"
    BACK_ENTRANCE = "back_entrance"
    COMMON_LAWN = "common_lawn"
    GREETINGS = "greetings"
    INTENT_CLARIFY = "intent_clarify"


class IntentClassification(BaseModel):
    """The classifier's complete, deliberately small output contract."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1, max_length=300)
    intent: IntentLabel

    @field_validator("reason")
    @classmethod
    def enforce_reason_token_limit(cls, reason: str) -> str:
        """Bound the user-visible explanation to at most 30 whitespace tokens."""
        normalized = " ".join(reason.split())
        if not normalized or len(normalized.split()) > 30:
            raise ValueError("Classification reason must contain 1 to 30 tokens")
        return normalized
