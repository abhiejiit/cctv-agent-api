"""Provider-backed direct LLM call for intent classification."""

from __future__ import annotations

import logging
import time

import httpx

from app.config import CameraCatalog, LLMConfig
from app.planning.models import IntentClassification, IntentLabel
from app.planning.policy import build_system_prompt
from app.planning.validation import parse_intent_classification
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
    ) -> None:
        self._client = client
        self._config = config
        self._system_prompt = build_system_prompt(catalog)

    async def classify(self, message: str) -> IntentClassification:
        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": message},
            ],
            "format": IntentClassification.model_json_schema(),
            "stream": False,
            "keep_alive": self._config.keep_alive,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_output_tokens,
            },
        }
        with tracer.start_as_current_span("cctv.llm.classify") as span:
            span.set_attribute("llm.provider", self._config.provider)
            span.set_attribute("llm.model", self._config.model)
            span.set_attribute("llm.request.timeout_seconds", self._config.timeout_seconds)
            started_at = time.perf_counter()
            logger.debug("LLM classification started model=%s", self._config.model)
            try:
                response = await self._client.post(
                    f"{str(self._config.base_url).rstrip('/')}{self._config.chat_path}",
                    json=payload,
                    timeout=self._config.timeout_seconds,
                )
                response.raise_for_status()
                content = response.json()["message"]["content"]
                result = parse_intent_classification(content)
                elapsed_ms = round((time.perf_counter() - started_at) * 1000)
                span.set_attribute("llm.response.intent", result.intent.value)
                span.set_attribute("llm.response.reason_tokens", len(result.reason.split()))
                span.set_attribute("llm.response.elapsed_ms", elapsed_ms)
                logger.info(
                    "LLM classification completed model=%s intent=%s elapsed_ms=%d",
                    self._config.model,
                    result.intent.value,
                    elapsed_ms,
                )
                logger.debug("LLM classification reason=%s", result.reason)
                return result
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                elapsed_ms = round((time.perf_counter() - started_at) * 1000)
                span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("llm.response.elapsed_ms", elapsed_ms)
                logger.warning(
                    "LLM classification failed model=%s elapsed_ms=%d error_type=%s",
                    self._config.model,
                    elapsed_ms,
                    type(exc).__name__,
                )
                raise LLMClassifierError("Intent classification is temporarily unavailable") from exc
