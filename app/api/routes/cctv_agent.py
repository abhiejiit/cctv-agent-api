"""Authenticated endpoint for submitting a CCTV assistant request."""

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from opentelemetry import trace
from opentelemetry.trace import Status as SpanStatus, StatusCode

from app.api.schemas import AskRequest, AskResponse, ObservationMetadata
from app.auth.middleware import get_principal
from app.auth.principal import Principal
from app.security.rate_limits import InMemoryRateLimiter
from app.workflow.request_flow import RequestWorkflow
from app.workflow.request_state import RequestState


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/cctv-agent", response_model=AskResponse)
async def ask_cctv_agent(
    body: AskRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> AskResponse:
    current_span = trace.get_current_span()
    settings = request.app.state.settings
    metrics = request.app.state.metrics
    request_started = time.perf_counter()
    current_span.set_attribute("cctv.request_id", request.state.request_id)
    logger.info("CCTV request received request_id=%s", request.state.request_id)
    max_message_length = min(
        settings.api.max_message_characters,
        settings.guardrails.max_message_characters,
    )
    if len(body.message) > max_message_length:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The message exceeds the configured length limit.",
        )
    limiter: InMemoryRateLimiter = request.app.state.rate_limiter
    try:
        await limiter.check(principal.user_id)
    except HTTPException as exc:
        if exc.status_code in {
            status.HTTP_429_TOO_MANY_REQUESTS,
            status.HTTP_503_SERVICE_UNAVAILABLE,
        }:
            if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                metrics.observe_rate_limit()
            metrics.observe_request("rejected")
            metrics.observe_stage("request", "rejected", time.perf_counter() - request_started)
        raise

    semaphore: asyncio.Semaphore = request.app.state.request_semaphore
    time_remaining = settings.server.request_timeout_seconds - (
        time.perf_counter() - request_started
    )
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=min(settings.api.admission_timeout_seconds, max(time_remaining, 0.001)),
        )
    except TimeoutError as exc:
        elapsed = time.perf_counter() - request_started
        deadline_exceeded = elapsed >= settings.server.request_timeout_seconds
        outcome = "timeout" if deadline_exceeded else "rejected"
        metrics.observe_request(outcome)
        metrics.observe_stage("request", outcome, elapsed)
        if deadline_exceeded:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "message": "The request exceeded the processing time limit.",
                    "request_id": request.state.request_id,
                },
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The service is busy. Please retry shortly.",
            headers={"Retry-After": "1"},
        ) from exc

    try:
        state = RequestState(
            request_id=request.state.request_id,
            principal=principal,
            user_message=body.message,
        )
        workflow: RequestWorkflow = request.app.state.request_workflow
        try:
            remaining = settings.server.request_timeout_seconds - (
                time.perf_counter() - request_started
            )
            async with asyncio.timeout(max(remaining, 0.001)):
                result = await workflow.run(state)
        except TimeoutError as exc:
            current_span.set_status(SpanStatus(StatusCode.ERROR, "request timeout"))
            current_span.set_attribute("cctv.workflow_status", "request_timeout")
            metrics.observe_request("timeout")
            metrics.observe_stage("request", "timeout", time.perf_counter() - request_started)
            logger.warning("CCTV request timed out request_id=%s", request.state.request_id)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "message": "The request exceeded the processing time limit.",
                    "request_id": request.state.request_id,
                },
            ) from exc

        current_span.set_attribute("cctv.intent", result.intent.value)
        current_span.set_attribute("cctv.workflow_status", result.workflow_status)
        if result.workflow_status.startswith("guardrail_blocked"):
            outcome = "blocked"
        elif result.workflow_status in {"guardrail_unavailable", "classification_fallback", "agent_unavailable"}:
            outcome = "error"
        else:
            outcome = "ok"
        metrics.observe_request(outcome)
        metrics.observe_stage("request", outcome, time.perf_counter() - request_started)

        answer = result.agent_response or "I could not prepare a response. Please try again."
        observation = None
        if result.observation_metadata:
            observation = ObservationMetadata(
                source=result.observation_metadata["source"],
                data_status=result.observation_metadata["data_status"],
                captured_at=result.observation_metadata["captured_at"],
            )
            if (
                observation.data_status == "mock"
                and not answer.startswith("Demo data only (mock snapshot, not live footage):")
            ):
                answer = f"Demo data only (mock snapshot, not live footage): {answer}"

        response_body = AskResponse(
            request_id=result.request_id,
            intent=result.intent,
            answer=answer,
            reason=result.classification_reason or "Please clarify what camera location you mean.",
            status=result.workflow_status,
            guardrail_status=result.guardrail_status,
            guardrail_findings=result.guardrail_findings,
            guardrail_topics=result.guardrail_topics,
            classification_source=result.classification_source,
            observation=observation,
        )
        if result.workflow_status == "agent_unavailable":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response_body.model_dump(mode="json"),
                headers={"Retry-After": "1"},
            )
        return response_body
    finally:
        semaphore.release()
