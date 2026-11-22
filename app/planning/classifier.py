"""Provider-backed direct LLM call for intent classification."""

from __future__ import annotations

import logging
import json
import time
from collections.abc import Sequence

import httpx

from app.config import CameraCatalog, LLMConfig, TracingConfig
from app.planning.models import IntentClassification, IntentLabel
from app.planning.policy import build_system_prompt
from app.planning.validation import parse_intent_classification
from app.services.conversation_history_service import ConversationTurn
from app.llm_provider import build_request_payload, parse_response, request_headers, request_url
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class LLMClassifierError(RuntimeError):
    """The configured LLM provider could not provide a valid classification."""


class LLMIntentClassifier:
    def __init__(
        self,
        client: httpx.AsyncClient,
        config: LLMConfig,
        catalog: CameraCatalog,
        tracing_config: TracingConfig,
    ) -> None:
        self._client = client
        self._config = config
        self._tracing_config = tracing_config
        self._system_prompt = build_system_prompt(catalog)

    async def classify(
        self,
        message: str,
        conversation_history: Sequence[ConversationTurn] = (),
    ) -> IntentClassification:
        contextual_request = json.dumps(
            {
                "conversation_history": [
                    {
                        "user_message": turn.user_message,
                        "agent_message": turn.agent_message,
                    }
                    for turn in conversation_history
                ],
                "current_message": message,
            },
            ensure_ascii=False,
        )
        payload = build_request_payload(
            self._config,
            system_prompt=self._system_prompt,
            user_message=contextual_request,
            response_schema=IntentClassification.model_json_schema(),
        )
        with tracer.start_as_current_span("cctv.llm.classify") as span:
            span.set_attribute("llm.provider", self._config.provider)
            span.set_attribute("llm.model", self._config.model)
            span.set_attribute("llm.thinking_model", self._config.thinking_model)
            span.set_attribute("llm.think", self._config.think)
            span.set_attribute("llm.request.timeout_seconds", self._config.timeout_seconds)
            span.set_attribute("llm.conversation_history.turn_count", len(conversation_history))
            if self._tracing_config.capture_llm_prompts:
                span.set_attribute("llm.prompt.system", self._system_prompt)
                span.set_attribute("llm.prompt.user", contextual_request)
            started_at = time.perf_counter()
            response_data: dict = {}
            logger.debug("LLM classification started model=%s", self._config.model)
            try:
                response = await self._client.post(
                    request_url(self._config),
                    headers=request_headers(self._config),
                    json=payload,
                    timeout=self._config.timeout_seconds,
                )
                response.raise_for_status()
                parsed_response = response.json()
                if not isinstance(parsed_response, dict):
                    raise TypeError("LLM provider response must be a JSON object")
                response_data = parsed_response
                normalized_response = parse_response(self._config, response_data)
                content = normalized_response.content
                thinking = normalized_response.thinking
                if thinking:
                    span.set_attribute("llm.response.thinking_present", True)
                    logger.debug(
                        "LLM classification returned model thinking model=%s characters=%d",
                        self._config.model,
                        len(thinking),
                    )
                if self._tracing_config.capture_llm_outputs:
                    span.set_attribute("llm.response.structured", str(content))
                if not content or not content.strip():
                    logger.warning(
                        "LLM classification returned empty structured content model=%s thinking_chars=%d; set llm.think=false for JSON classification",
                        self._config.model,
                        len(thinking),
                    )
                result = parse_intent_classification(content)
                elapsed_ms = round((time.perf_counter() - started_at) * 1000)
                done_reason = normalized_response.finish_reason
                output_tokens = normalized_response.output_tokens
                span.set_attribute("llm.response.intent", result.intent.value)
                if self._tracing_config.capture_llm_outputs:
                    span.set_attribute("llm.response.reason", result.reason)
                span.set_attribute("llm.response.reason_tokens", len(result.reason.split()))
                span.set_attribute("llm.response.elapsed_ms", elapsed_ms)
                span.set_attribute("llm.response.done_reason", str(done_reason))
                if output_tokens is not None:
                    span.set_attribute("llm.response.output_tokens", int(output_tokens))
                logger.info(
                    "LLM classification completed model=%s intent=%s elapsed_ms=%d done_reason=%s output_tokens=%s",
                    self._config.model,
                    result.intent.value,
                    elapsed_ms,
                    done_reason,
                    output_tokens if output_tokens is not None else "unknown",
                )
                if self._tracing_config.capture_llm_outputs:
                    logger.debug("LLM classification reason=%s", result.reason)
                return result
            except httpx.HTTPStatusError as exc:
                elapsed_ms = round((time.perf_counter() - started_at) * 1000)
                span.set_status(Status(StatusCode.ERROR, "llm http error"))
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("llm.response.status_code", exc.response.status_code)
                logger.error(
                    "LLM classification provider rejected model=%s status=%d",
                    self._config.model,
                    exc.response.status_code,
                )
                raise LLMClassifierError("Intent classification is temporarily unavailable") from exc
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                elapsed_ms = round((time.perf_counter() - started_at) * 1000)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("llm.response.elapsed_ms", elapsed_ms)
                try:
                    normalized_response = parse_response(self._config, response_data)
                    done_reason = normalized_response.finish_reason
                    output_tokens = normalized_response.output_tokens or "unknown"
                except (KeyError, TypeError, ValueError):
                    done_reason = "unknown"
                    output_tokens = "unknown"
                span.set_attribute("llm.response.done_reason", str(done_reason))
                logger.warning(
                    "LLM classification failed model=%s elapsed_ms=%d error_type=%s done_reason=%s output_tokens=%s",
                    self._config.model,
                    elapsed_ms,
                    type(exc).__name__,
                    done_reason,
                    output_tokens,
                )
                raise LLMClassifierError("Intent classification is temporarily unavailable") from exc
