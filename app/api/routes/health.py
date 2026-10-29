"""Liveness and dependency-readiness endpoints."""

import logging

import httpx
from fastapi import APIRouter, Request, Response, status

from app.api.schemas import HealthResponse


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request, response: Response) -> HealthResponse:
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client
    try:
        result = await client.get(
            f"{str(settings.llm.base_url).rstrip('/')}{settings.llm.readiness_path}",
            timeout=min(settings.llm.timeout_seconds, 3.0),
        )
        result.raise_for_status()
    except httpx.HTTPError:
        logger.warning("LLM readiness check failed provider=%s", settings.llm.provider)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", llm="unreachable")
    logger.debug("LLM readiness check succeeded provider=%s", settings.llm.provider)
    return HealthResponse(status="ok", llm="reachable")
