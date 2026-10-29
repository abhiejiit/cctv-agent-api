"""Authenticated endpoint for submitting a CCTV assistant request."""

import logging

from fastapi import APIRouter, Depends, Request

from app.api.schemas import AskRequest, AskResponse
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
    logger.info("CCTV request received request_id=%s user_id=%s", request.state.request_id, principal.user_id)
    limiter: InMemoryRateLimiter = request.app.state.rate_limiter
    await limiter.check(principal.user_id)

    state = RequestState(
        request_id=request.state.request_id,
        principal=principal,
        user_message=body.message,
    )
    workflow: RequestWorkflow = request.app.state.request_workflow
    result = await workflow.run(state)

    return AskResponse(
        request_id=result.request_id,
        intent=result.intent,
        reason=result.classification_reason or "Please clarify what camera location you mean.",
        status=result.workflow_status,
        guardrail_status=result.guardrail_status,
        guardrail_findings=result.guardrail_findings,
        classification_source=result.classification_source,
    )
