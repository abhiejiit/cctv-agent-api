"""Validation helpers for classifier output."""

from app.planning.models import IntentClassification


def parse_intent_classification(content: str) -> IntentClassification:
    """Accept only schema-valid JSON from the configured local model."""
    return IntentClassification.model_validate_json(content)
