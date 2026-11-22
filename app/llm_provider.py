"""Provider adapters for direct LLM requests and CrewAI model construction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from app.config import LLMConfig


GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


@dataclass(frozen=True)
class LLMResponse:
    content: str
    thinking: str = ""
    finish_reason: str = "unknown"
    output_tokens: int | None = None


class LLMProviderAdapter(Protocol):
    """Operations that vary between an LLM provider's APIs."""

    name: str

    def request_url(self, config: LLMConfig) -> str: ...

    def readiness_url(self, config: LLMConfig) -> str: ...

    def readiness_credential_key(self, config: LLMConfig) -> str: ...

    def request_headers(self, config: LLMConfig) -> dict[str, str]: ...

    def build_request_payload(
        self,
        config: LLMConfig,
        *,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any],
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> dict[str, Any]: ...

    def parse_response(self, provider_response: dict[str, Any]) -> LLMResponse: ...

    def available_models(self, provider_response: dict[str, Any]) -> set[str]: ...

    def configured_model_name(self, config: LLMConfig) -> str: ...

    def crewai_model_options(self, config: LLMConfig) -> tuple[str, dict[str, Any]]: ...


class ProviderRegistry:
    """Named provider adapter registry, extensible without router conditionals."""

    def __init__(self) -> None:
        self._adapters: dict[str, LLMProviderAdapter] = {}

    def register(self, adapter: LLMProviderAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f"LLM provider adapter already registered: {adapter.name}")
        self._adapters[adapter.name] = adapter

    def get(self, provider_name: str) -> LLMProviderAdapter:
        try:
            return self._adapters[provider_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported LLM provider: {provider_name}") from exc


class OllamaAdapter:
    name = "ollama"

    def request_url(self, config: LLMConfig) -> str:
        return f"{str(config.base_url).rstrip('/')}{config.chat_path}"

    def readiness_url(self, config: LLMConfig) -> str:
        return f"{str(config.base_url).rstrip('/')}{config.readiness_path}"

    def readiness_credential_key(self, config: LLMConfig) -> str:
        return ""

    def request_headers(self, config: LLMConfig) -> dict[str, str]:
        return {}

    def build_request_payload(
        self,
        config: LLMConfig,
        *,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any],
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        configured_temperature = config.temperature if temperature is None else temperature
        configured_max_tokens = (
            config.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "format": response_schema,
            "stream": False,
            "keep_alive": config.keep_alive,
            "options": {
                "temperature": configured_temperature,
                "num_predict": configured_max_tokens,
            },
        }
        if config.thinking_model:
            payload["think"] = config.think
        return payload

    def parse_response(self, provider_response: dict[str, Any]) -> LLMResponse:
        message = provider_response["message"]
        return LLMResponse(
            content=message["content"],
            thinking=message.get("thinking", ""),
            finish_reason=str(provider_response.get("done_reason", "unknown")),
            output_tokens=_optional_int(provider_response.get("eval_count")),
        )

    def available_models(self, provider_response: dict[str, Any]) -> set[str]:
        return {
            name
            for item in _model_items(provider_response)
            for name in (item.get("name"), item.get("model"))
            if isinstance(name, str)
        }

    def configured_model_name(self, config: LLMConfig) -> str:
        return config.model

    def crewai_model_options(self, config: LLMConfig) -> tuple[str, dict[str, Any]]:
        return (
            f"ollama_chat/{config.model}",
            {
                "base_url": str(config.base_url).rstrip("/"),
                **_crewai_generation_options(config),
            },
        )


class GeminiAdapter:
    name = "gemini"

    def request_url(self, config: LLMConfig) -> str:
        model = normalize_model_name(config.model)
        return f"{provider_base_url(config)}/models/{quote(model, safe='.-')}:generateContent"

    def readiness_url(self, config: LLMConfig) -> str:
        return f"{provider_base_url(config)}/models"

    def readiness_credential_key(self, config: LLMConfig) -> str:
        return config.api_key_environment_variable

    def request_headers(self, config: LLMConfig) -> dict[str, str]:
        api_key = self._api_key(config)
        return {"Content-Type": "application/json", "x-goog-api-key": api_key}

    def build_request_payload(
        self,
        config: LLMConfig,
        *,
        system_prompt: str,
        user_message: str,
        response_schema: dict[str, Any],
        temperature: float | None,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        configured_temperature = config.temperature if temperature is None else temperature
        configured_max_tokens = (
            config.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": configured_temperature,
                "maxOutputTokens": configured_max_tokens,
                "responseMimeType": "application/json",
                "responseJsonSchema": _gemini_schema(response_schema),
            },
        }

    def parse_response(self, provider_response: dict[str, Any]) -> LLMResponse:
        candidates = provider_response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini response did not include a candidate")
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise TypeError("Gemini response candidate must be an object")
        content = candidate.get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        text_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and not part.get("thought")
            and isinstance(part.get("text"), str)
        ]
        thought_parts = [
            part["text"]
            for part in parts
            if isinstance(part, dict)
            and part.get("thought")
            and isinstance(part.get("text"), str)
        ]
        usage = provider_response.get("usageMetadata", {})
        output_tokens = usage.get("candidatesTokenCount") if isinstance(usage, dict) else None
        return LLMResponse(
            content="".join(text_parts),
            thinking="\n".join(thought_parts),
            finish_reason=str(candidate.get("finishReason", "unknown")),
            output_tokens=_optional_int(output_tokens),
        )

    def available_models(self, provider_response: dict[str, Any]) -> set[str]:
        return {
            normalize_model_name(name)
            for item in _model_items(provider_response)
            if isinstance(item.get("supportedGenerationMethods", []), list)
            and "generateContent" in item.get("supportedGenerationMethods", [])
            for name in (item.get("name"),)
            if isinstance(name, str)
        }

    def configured_model_name(self, config: LLMConfig) -> str:
        return normalize_model_name(config.model)

    def crewai_model_options(self, config: LLMConfig) -> tuple[str, dict[str, Any]]:
        return (
            f"gemini/{normalize_model_name(config.model)}",
            {"api_key": self._api_key(config), **_crewai_generation_options(config)},
        )

    @staticmethod
    def _api_key(config: LLMConfig) -> str:
        api_key = os.getenv(config.api_key_environment_variable, "").strip()
        if not api_key or api_key.casefold() in {
            "your_api_key", "<your-api-key>", "your-api-key", "changeme"
        }:
            raise ValueError(
                "Gemini API key is unavailable to this process. Set "
                f"{config.api_key_environment_variable} in the environment used to start the API."
            )
        return api_key


PROVIDERS = ProviderRegistry()
PROVIDERS.register(OllamaAdapter())
PROVIDERS.register(GeminiAdapter())


def register_provider(adapter: LLMProviderAdapter) -> None:
    """Register another adapter during application bootstrap or in an extension."""
    PROVIDERS.register(adapter)


def get_provider(provider_name: str) -> LLMProviderAdapter:
    return PROVIDERS.get(provider_name)


def create_crewai_llm(config: LLMConfig) -> Any:
    """Create a CrewAI LLM through the provider-specific registered adapter."""
    model, options = get_provider(config.provider).crewai_model_options(config)
    from crewai import LLM

    return LLM(model=model, **options)


def request_url(config: LLMConfig) -> str:
    return get_provider(config.provider).request_url(config)


def readiness_url(config: LLMConfig) -> str:
    return get_provider(config.provider).readiness_url(config)


def request_headers(config: LLMConfig) -> dict[str, str]:
    return get_provider(config.provider).request_headers(config)


def build_request_payload(
    config: LLMConfig,
    *,
    system_prompt: str,
    user_message: str,
    response_schema: dict[str, Any],
    temperature: float | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    return get_provider(config.provider).build_request_payload(
        config,
        system_prompt=system_prompt,
        user_message=user_message,
        response_schema=response_schema,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def parse_response(config: LLMConfig, provider_response: dict[str, Any]) -> LLMResponse:
    return get_provider(config.provider).parse_response(provider_response)


def normalize_model_name(model: str) -> str:
    return model.removeprefix("models/")


def provider_base_url(config: LLMConfig) -> str:
    configured_base_url = str(config.base_url).rstrip("/")
    if config.provider == "gemini" and configured_base_url == "http://localhost:11434":
        return GEMINI_DEFAULT_BASE_URL
    return configured_base_url


def _crewai_generation_options(config: LLMConfig) -> dict[str, Any]:
    return {
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
        "timeout": config.timeout_seconds,
    }


def _model_items(provider_response: dict[str, Any]) -> list[dict[str, Any]]:
    models = provider_response.get("models")
    if not isinstance(models, list):
        raise ValueError("Invalid model readiness response")
    return [item for item in models if isinstance(item, dict)]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local Pydantic refs and retain Gemini-supported JSON Schema keys."""
    definitions = schema.get("$defs", {})
    supported_keys = {
        "type", "properties", "required", "items", "enum", "description",
        "format", "minimum", "maximum", "minItems", "maxItems", "minLength",
        "maxLength", "pattern", "anyOf", "nullable",
    }

    def visit(node: Any) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node
        reference = node.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.rsplit("/", 1)[-1]
            if name not in definitions:
                raise ValueError("Unknown local schema reference")
            return visit(definitions[name])
        converted: dict[str, Any] = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                converted[key] = {
                    property_name: visit(property_schema)
                    for property_name, property_schema in value.items()
                }
            elif key in supported_keys:
                converted[key] = visit(value)
        return converted

    converted = visit(schema)
    if not isinstance(converted, dict) or not converted.get("type"):
        raise ValueError("Gemini response schema must define an object type")
    return converted
