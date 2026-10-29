"""Canonicalize user input before deterministic guardrail inspection."""

import unicodedata

import logging

logger = logging.getLogger(__name__)

def normalize_user_input(message: str) -> str:
    """Apply Unicode normalization and remove control characters except whitespace."""
    normalized = unicodedata.normalize("NFKC", message)
    normalized_text="".join(
        character
        for character in normalized
        if character in "\n\r\t" or not unicodedata.category(character).startswith("C")
    ).strip()
    logger.info(f"The normalized text from input guardrails: {normalized_text}")
    return normalized_text
