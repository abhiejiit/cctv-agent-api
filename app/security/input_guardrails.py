"""Fast, deterministic input limits and configurable pattern checks."""

import re
from dataclasses import dataclass
from typing import Literal

from app.config import GuardrailConfig
from app.security.input_validation import normalize_user_input


GuardrailStatus = Literal["allowed", "blocked"]


@dataclass(frozen=True)
class GuardrailResult:
    status: GuardrailStatus
    normalized_message: str
    findings: tuple[str, ...]


class InputGuardrails:
    def __init__(self, config: GuardrailConfig) -> None:
        self._config = config
        self._patterns = tuple(
            (item.category, re.compile(item.pattern)) for item in config.blocked_patterns
        )

    def inspect(self, message: str) -> GuardrailResult:
        normalized = normalize_user_input(message)
        findings: list[str] = []

        if len(normalized) > self._config.max_message_characters:
            findings.append("message_too_long")

        for category, pattern in self._patterns:
            if pattern.search(normalized):
                findings.append(category)

        unique_findings = tuple(dict.fromkeys(findings))
        return GuardrailResult(
            status="blocked" if unique_findings else "allowed",
            normalized_message=normalized,
            findings=unique_findings,
        )
