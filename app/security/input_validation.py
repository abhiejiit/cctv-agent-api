"""Canonicalize user input before deterministic guardrail inspection."""

import unicodedata


def normalize_user_input(message: str) -> str:
    """Apply Unicode normalization and remove control characters except whitespace."""
    normalized = unicodedata.normalize("NFKC", message)
    normalized_text = "".join(
        character
        for character in normalized
        if character in "\n\r\t" or not unicodedata.category(character).startswith("C")
    ).strip()
    return normalized_text
