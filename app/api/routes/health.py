"""Liveness and dependency-readiness endpoints."""

import logging
import hmac
import os

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, status

from app.api.schemas import HealthResponse
from app.config import LLMConfig


router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(request: Request, response: Response) -> HealthResponse:
    settings = request.app.state.settings
    client: httpx.AsyncClient = request.app.state.http_client
    configs = [settings.llm]
    guardrail_enabled = settings.guardrails.llm_filter.enabled
    if guardrail_enabled:
        configs.append(settings.guardrail_llm)

    endpoint_results: dict[str, tuple[str, set[str]]] = {}
    for config in configs:
        endpoint = f"{str(config.base_url).rstrip('/')}{config.readiness_path}"
        if endpoint in endpoint_results:
            continue
        try:
            result = await client.get(endpoint, timeout=min(config.timeout_seconds, 3.0))
            result.raise_for_status()
            model_payload = result.json()
            if not isinstance(model_payload, dict) or not isinstance(model_payload.get("models"), list):
                raise ValueError("Invalid model readiness response")
            model_names = {
                name
                for item in model_payload["models"]
                if isinstance(item, dict)
                for name in (item.get("name"), item.get("model"))
                if isinstance(name, str)
            }
            endpoint_results[endpoint] = ("reachable", model_names)
        except (httpx.HTTPError, ValueError):
            endpoint_results[endpoint] = ("unreachable", set())

    def model_status(config: LLMConfig) -> str:
        endpoint = f"{str(config.base_url).rstrip('/')}{config.readiness_path}"
        endpoint_status, model_names = endpoint_results[endpoint]
        if endpoint_status != "reachable":
            return "unreachable"
        configured_model = config.model
        if configured_model not in model_names and f"{configured_model}:latest" not in model_names:
            return "model_unavailable"
        return "reachable"

    llm_status = model_status(settings.llm)
    guardrail_status = model_status(settings.guardrail_llm) if guardrail_enabled else "disabled"
    if llm_status != "reachable" or guardrail_status not in {"reachable", "disabled"}:
        logger.warning("LLM readiness check failed llm=%s guardrail=%s", llm_status, guardrail_status)
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unavailable",
            llm=llm_status,
            guardrail_llm=guardrail_status,
        )
    logger.debug("LLM readiness check succeeded")
    return HealthResponse(status="ok", llm=llm_status, guardrail_llm=guardrail_status)


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    settings = request.app.state.settings
    if not settings.metrics.enabled:
        raise HTTPException(status_code=404, detail="Not found")
    token_variable = settings.metrics.auth_token_environment_variable
    if settings.application.environment == "production" and not token_variable:
        raise HTTPException(status_code=404, detail="Not found")
    if token_variable:
        expected_token = os.getenv(token_variable, "")
        scheme, _, supplied_token = request.headers.get("Authorization", "").partition(" ")
        if not expected_token:
            raise HTTPException(status_code=503, detail="Metrics authentication is unavailable")
        if scheme.lower() != "bearer" or not hmac.compare_digest(supplied_token, expected_token):
            raise HTTPException(
                status_code=401,
                detail="A valid bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return Response(
        content=request.app.state.metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
