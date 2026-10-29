"""Request flow through state initialization, guardrails, and intent classification."""

import logging

from app.planning.classifier import LLMClassifierError, LLMIntentClassifier
from app.planning.models import IntentLabel
from app.security.input_guardrails import InputGuardrails
from app.workflow.request_state import RequestState


logger = logging.getLogger(__name__)


class RequestWorkflow:
    def __init__(
        self,
        guardrails: InputGuardrails,
        classifier: LLMIntentClassifier,
    ) -> None:
        self._guardrails = guardrails
        self._classifier = classifier

    async def run(self, state: RequestState) -> RequestState:
        """Inspect input, record findings, then classify into a closed intent set."""
        logger.info("Request workflow started request_id=%s", state.request_id)
        guardrail_result = self._guardrails.inspect(state.user_message)
        state.normalized_message = guardrail_result.normalized_message
        state.guardrail_status = guardrail_result.status
        state.guardrail_findings = list(guardrail_result.findings)
        logger.info(
            "Input guardrails completed request_id=%s status=%s findings=%s",
            state.request_id,
            state.guardrail_status,
            ",".join(state.guardrail_findings) or "none",
        )

        # Never send a detected malicious/oversized message to the LLM. The
        # configured classifier contract still resolves this safely to clarify.
        if guardrail_result.status == "blocked":
            state.intent = IntentLabel.INTENT_CLARIFY
            state.classification_reason = (
                "I could not safely process that request. Please rephrase it as a camera question."
            )
            state.classification_source = "guardrail"
            state.workflow_status = "guardrail_blocked"
            logger.warning("Request blocked by guardrails request_id=%s", state.request_id)
            return state

        try:
            classification = await self._classifier.classify(guardrail_result.normalized_message)
            state.intent = classification.intent
            state.classification_reason = classification.reason
            state.classification_source = "llm"
            state.workflow_status = "classified"
        except LLMClassifierError:
            # Fail closed to a clarification path; do not invent a camera route.
            state.intent = IntentLabel.INTENT_CLARIFY
            state.classification_reason = (
                "I could not determine the right camera. Please try again or clarify the location."
            )
            state.classification_source = "fallback"
            state.workflow_status = "classification_fallback"
            logger.warning("Request used classifier fallback request_id=%s", state.request_id)
        logger.info(
            "Request workflow completed request_id=%s status=%s intent=%s source=%s",
            state.request_id,
            state.workflow_status,
            state.intent.value if state.intent else "unset",
            state.classification_source,
        )
        return state
