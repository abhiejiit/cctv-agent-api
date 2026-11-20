"""Request flow through guardrails, classification, and selected CrewAI agent."""

import logging
import json
import time

from app.config import CameraCatalog, TracingConfig
from app.orchestration.intent_router import CrewAIExecutionError, IntentAgentRouter
from app.planning.classifier import LLMClassifierError, LLMIntentClassifier
from app.planning.models import IntentLabel
from app.security.input_guardrails import InputGuardrails
from app.security.llm_input_filter import LLMInputFilter, LLMInputGuardrailError
from app.services.conversation_history_service import ConversationHistoryService
from app.telemetry.metrics import AppMetrics
from app.workflow.request_state import RequestState
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode


logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
GUARDRAIL_END_MESSAGE = "we're sorry, we cannot proceed with that"
_MAX_HISTORY_TRACE_CHARACTERS = 24_000


def _record_history_span(span: Span, state: RequestState, capture_content: bool) -> None:
    """Record queue metadata, optionally including the readable queue contents."""
    turns = state.conversation_history
    span.set_attribute("cctv.conversation_history.status", state.conversation_history_status)
    span.set_attribute("cctv.conversation_history.turn_count", len(turns))
    span.set_attribute("cctv.conversation_history.capture_content", capture_content)
    if not capture_content:
        return

    queue_json = json.dumps(
        [turn.model_dump(mode="json") for turn in turns],
        ensure_ascii=False,
        indent=2,
    )
    truncated = len(queue_json) > _MAX_HISTORY_TRACE_CHARACTERS
    if truncated:
        queue_json = queue_json[:_MAX_HISTORY_TRACE_CHARACTERS] + "\n...[truncated]"
    span.set_attribute("cctv.conversation_history.queue", queue_json)
    span.set_attribute("cctv.conversation_history.queue_truncated", truncated)


class RequestWorkflow:
    def __init__(
        self,
        guardrails: InputGuardrails,
        llm_input_filter: LLMInputFilter,
        classifier: LLMIntentClassifier,
        agent_router: IntentAgentRouter,
        camera_catalog: CameraCatalog,
        metrics: AppMetrics,
        conversation_history_service: ConversationHistoryService | None = None,
        tracing_config: TracingConfig | None = None,
    ) -> None:
        self._guardrails = guardrails
        self._llm_input_filter = llm_input_filter
        self._classifier = classifier
        self._agent_router = agent_router
        self._camera_catalog = camera_catalog
        self._metrics = metrics
        self._conversation_history_service = conversation_history_service
        self._tracing_config = tracing_config or TracingConfig()

    async def run(self, state: RequestState) -> RequestState:
        """Inspect input, record findings, then classify into a closed intent set."""
        with tracer.start_as_current_span("cctv.request.workflow") as workflow_span:
            workflow_span.set_attribute("cctv.request_id", state.request_id)
            logger.info("Request workflow started request_id=%s", state.request_id)
            if self._conversation_history_service is None:
                state.conversation_history_status = "disabled"
                _record_history_span(
                    workflow_span, state, self._tracing_config.capture_conversation_history
                )
            else:
                history_started = time.perf_counter()
                with tracer.start_as_current_span("cctv.conversation_history.load") as history_span:
                    try:
                        state.conversation_history = (
                            await self._conversation_history_service.get_recent_turns(state.principal)
                        )
                        state.conversation_history_status = "available"
                        _record_history_span(
                            history_span, state, self._tracing_config.capture_conversation_history
                        )
                        _record_history_span(
                            workflow_span, state, self._tracing_config.capture_conversation_history
                        )
                        self._metrics.observe_stage(
                            "conversation_history", "ok", time.perf_counter() - history_started
                        )
                    except Exception as exc:
                        state.conversation_history = []
                        state.conversation_history_status = "unavailable"
                        history_span.set_attribute("error.type", type(exc).__name__)
                        _record_history_span(
                            history_span, state, self._tracing_config.capture_conversation_history
                        )
                        _record_history_span(
                            workflow_span, state, self._tracing_config.capture_conversation_history
                        )
                        self._metrics.observe_stage(
                            "conversation_history", "error", time.perf_counter() - history_started
                        )
                        logger.warning(
                            "Conversation history unavailable operation=load error_type=%s",
                            type(exc).__name__,
                        )
            with tracer.start_as_current_span("cctv.input_guardrails") as guardrail_span:
                stage_started = time.perf_counter()
                guardrail_result = self._guardrails.inspect(state.user_message)
                self._metrics.observe_stage(
                    "guardrails",
                    "blocked" if guardrail_result.status == "blocked" else "ok",
                    time.perf_counter() - stage_started,
                )
                state.normalized_message = guardrail_result.normalized_message
                state.guardrail_status = guardrail_result.status
                state.guardrail_findings = list(guardrail_result.findings)
                guardrail_span.set_attribute("cctv.guardrail.status", state.guardrail_status)
                guardrail_span.set_attribute(
                    "cctv.guardrail.findings", state.guardrail_findings
                )
            logger.info(
                "Input guardrails completed request_id=%s status=%s findings=%s",
                state.request_id,
                state.guardrail_status,
                ",".join(state.guardrail_findings) or "none",
            )

            # Never send a detected malicious/oversized message to the LLM.
            if guardrail_result.status == "blocked":
                state.intent = IntentLabel.INTENT_CLARIFY
                state.classification_reason = (
                    "I could not safely process that request. Please rephrase it as a camera question."
                )
                state.classification_source = "guardrail"
                state.workflow_status = "guardrail_blocked"
                state.agent_response = GUARDRAIL_END_MESSAGE
                workflow_span.set_attribute("cctv.intent", state.intent.value)
                workflow_span.set_attribute("cctv.workflow_status", state.workflow_status)
                logger.warning("Request blocked by guardrails request_id=%s", state.request_id)
                return state

            if self._llm_input_filter.enabled:
                stage_started = time.perf_counter()
                try:
                    assessment = await self._llm_input_filter.assess(
                        guardrail_result.normalized_message
                    )
                except LLMInputGuardrailError:
                    self._metrics.observe_stage(
                        "guardrail_llm", "error", time.perf_counter() - stage_started
                    )
                    # Fail closed: do not run the intent classifier when policy
                    # evaluation did not complete reliably.
                    state.guardrail_status = "unavailable"
                    state.intent = IntentLabel.INTENT_CLARIFY
                    state.classification_reason = GUARDRAIL_END_MESSAGE
                    state.classification_source = "guardrail"
                    state.workflow_status = "guardrail_unavailable"
                    state.agent_response = GUARDRAIL_END_MESSAGE
                    workflow_span.set_status(Status(StatusCode.ERROR, "guardrail unavailable"))
                    workflow_span.set_attribute("cctv.guardrail.status", "unavailable")
                    workflow_span.set_attribute("cctv.intent", state.intent.value)
                    workflow_span.set_attribute("cctv.workflow_status", state.workflow_status)
                    logger.warning(
                        "Request stopped because LLM input guardrail is unavailable request_id=%s",
                        state.request_id,
                    )
                    return state

                self._metrics.observe_stage(
                    "guardrail_llm",
                    "blocked" if assessment.flagged else "ok",
                    time.perf_counter() - stage_started,
                )

                state.guardrail_topics = assessment.topics
                if assessment.flagged:
                    state.guardrail_status = "blocked"
                    state.intent = IntentLabel.INTENT_CLARIFY
                    state.classification_reason = GUARDRAIL_END_MESSAGE
                    state.classification_source = "guardrail"
                    state.workflow_status = "guardrail_blocked"
                    state.agent_response = GUARDRAIL_END_MESSAGE
                    workflow_span.set_attribute("cctv.guardrail.status", "blocked")
                    workflow_span.set_attribute("cctv.guardrail.topics", assessment.topics)
                    workflow_span.set_attribute("cctv.intent", state.intent.value)
                    workflow_span.set_attribute("cctv.workflow_status", state.workflow_status)
                    logger.info(
                        "Request blocked by LLM input guardrail request_id=%s topics=%s",
                        state.request_id,
                        ",".join(assessment.topics),
                    )
                    return state

            try:
                stage_started = time.perf_counter()
                with tracer.start_as_current_span("cctv.intent_classification"):
                    classification = await self._classifier.classify(
                        guardrail_result.normalized_message,
                        state.conversation_history,
                    )
                state.intent = classification.intent
                state.classification_reason = classification.reason
                state.classification_source = "llm"
                state.workflow_status = "classified"
            except LLMClassifierError:
                self._metrics.observe_stage(
                    "classifier", "error", time.perf_counter() - stage_started
                )
                # Fail closed to a clarification path; do not invent a camera route.
                state.intent = IntentLabel.INTENT_CLARIFY
                state.classification_reason = (
                    "I could not determine the right camera. Please try again or clarify the location."
                )
                state.classification_source = "fallback"
                state.workflow_status = "classification_fallback"
                workflow_span.set_status(Status(StatusCode.ERROR, "classification fallback"))
                logger.warning("Request used classifier fallback request_id=%s", state.request_id)
            else:
                self._metrics.observe_stage(
                    "classifier", "ok", time.perf_counter() - stage_started
                )
            workflow_span.set_attribute("cctv.intent", state.intent.value)
            workflow_span.set_attribute("cctv.workflow_status", state.workflow_status)
            stage_started = time.perf_counter()
            try:
                agent_result = await self._agent_router.invoke(
                    state.intent,
                    state.normalized_message or state.user_message,
                    self._camera_catalog,
                    state.principal.allowed_camera_ids,
                    state.conversation_history,
                )
            except CrewAIExecutionError:
                self._metrics.observe_stage("agent", "error", time.perf_counter() - stage_started)
                state.workflow_status = "agent_unavailable"
                state.agent_response = "The camera assistant is temporarily unavailable. Please try again."
                state.classification_reason = "The selected camera assistant could not complete this request."
                workflow_span.set_status(Status(StatusCode.ERROR, "agent unavailable"))
                workflow_span.set_attribute("cctv.workflow_status", state.workflow_status)
                logger.warning("Request agent unavailable request_id=%s", state.request_id)
                return state
            self._metrics.observe_stage("agent", "ok", time.perf_counter() - stage_started)
            state.agent_response = agent_result.answer
            state.observation_metadata = agent_result.observation_metadata
            if (
                state.observation_metadata
                and state.observation_metadata.get("data_status") == "mock"
                and not state.agent_response.startswith("Demo data only (mock snapshot, not live footage):")
            ):
                state.agent_response = (
                    "Demo data only (mock snapshot, not live footage): " + state.agent_response
                )
            workflow_span.set_attribute("cctv.agent_response.length", len(state.agent_response))
            if self._conversation_history_service is not None and state.guardrail_status == "allowed":
                history_started = time.perf_counter()
                with tracer.start_as_current_span("cctv.conversation_history.append") as history_span:
                    try:
                        state.conversation_history = await self._conversation_history_service.append_turn(
                            state.principal,
                            state.normalized_message or state.user_message,
                            state.agent_response,
                        )
                        state.conversation_history_status = "available"
                        self._metrics.observe_stage(
                            "conversation_history", "ok", time.perf_counter() - history_started
                        )
                    except Exception as exc:
                        state.conversation_history_status = "unavailable"
                        history_span.set_attribute("cctv.conversation_history.status", "unavailable")
                        history_span.set_attribute("error.type", type(exc).__name__)
                        self._metrics.observe_stage(
                            "conversation_history", "error", time.perf_counter() - history_started
                        )
                        logger.warning(
                            "Conversation history unavailable operation=append error_type=%s",
                            type(exc).__name__,
                        )
                    _record_history_span(
                        history_span, state, self._tracing_config.capture_conversation_history
                    )
                    _record_history_span(
                        workflow_span, state, self._tracing_config.capture_conversation_history
                    )
            logger.info(
                "Request workflow completed request_id=%s status=%s intent=%s source=%s",
                state.request_id,
                state.workflow_status,
                state.intent.value if state.intent else "unset",
                state.classification_source,
            )
            return state
