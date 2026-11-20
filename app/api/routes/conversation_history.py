"""Authenticated operations for the caller's conversation history."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request, status
from opentelemetry import trace
from opentelemetry.trace import Status as SpanStatus, StatusCode

from app.api.schemas import ClearConversationHistoryResponse
from app.auth.middleware import get_principal
from app.auth.principal import Principal


router = APIRouter()
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@router.post(
    "/conversation-history",
    response_model=ClearConversationHistoryResponse,
    summary="Clear the authenticated user's conversation history",
)
async def clear_conversation_history(
    request: Request,
    principal: Principal = Depends(get_principal),
) -> ClearConversationHistoryResponse:
    """Delete only the Redis conversation queue belonging to this principal."""
    history_service = getattr(request.app.state, "conversation_history_service", None)
    if history_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation history is unavailable.",
        )

    started_at = time.perf_counter()
    with tracer.start_as_current_span("cctv.conversation_history.clear") as span:
        span.set_attribute("cctv.request_id", request.state.request_id)
        try:
            await history_service.clear_history(principal)
        except Exception as exc:
            span.set_status(SpanStatus(StatusCode.ERROR, "conversation history clear failed"))
            span.set_attribute("cctv.conversation_history.status", "unavailable")
            span.set_attribute("error.type", type(exc).__name__)
            request.app.state.metrics.observe_stage(
                "conversation_history", "error", time.perf_counter() - started_at
            )
            logger.warning(
                "Conversation history clear failed request_id=%s error_type=%s",
                request.state.request_id,
                type(exc).__name__,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation history is temporarily unavailable.",
                headers={"Retry-After": "1"},
            ) from exc

        elapsed_seconds = time.perf_counter() - started_at
        span.set_attribute("cctv.conversation_history.status", "cleared")
        span.set_attribute("cctv.conversation_history.elapsed_ms", round(elapsed_seconds * 1000))
        request.app.state.metrics.observe_stage("conversation_history", "ok", elapsed_seconds)
        logger.info("Conversation history cleared request_id=%s", request.state.request_id)

    return ClearConversationHistoryResponse(
        status="cleared",
        request_id=request.state.request_id,
    )
