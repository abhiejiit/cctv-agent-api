"""Liveness and dependency-readiness endpoints."""

import logging
import hmac
import os

import httpx
from fastapi import APIRouter, HTTPException, Request, Response, status

from app.api.schemas import HealthResponse
from app.config import LLMConfig
from app.llm_provider import get_provider


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

    endpoint_results: dict[tuple[str, str, str], tuple[str, set[str]]] = {}
    for config in configs:
        adapter = get_provider(config.provider)
        endpoint = adapter.readiness_url(config)
        credential_id = adapter.readiness_credential_key(config)
        try:
            headers = adapter.request_headers(config)
        except ValueError:
            endpoint_results[(config.provider, endpoint, credential_id)] = ("unreachable", set())
            continue
        endpoint_key = (config.provider, endpoint, credential_id)
        if endpoint_key in endpoint_results:
            continue
        try:
            result = await client.get(
                endpoint,
                headers=headers,
                timeout=min(config.timeout_seconds, 3.0),
            )
            result.raise_for_status()
            model_payload = result.json()
            if not isinstance(model_payload, dict) or not isinstance(model_payload.get("models"), list):
                raise ValueError("Invalid model readiness response")
            model_names = adapter.available_models(model_payload)
            endpoint_results[endpoint_key] = ("reachable", model_names)
        except (httpx.HTTPError, ValueError):
            endpoint_results[endpoint_key] = ("unreachable", set())

    def model_status(config: LLMConfig) -> str:
        adapter = get_provider(config.provider)
        endpoint = adapter.readiness_url(config)
        credential_id = adapter.readiness_credential_key(config)
        endpoint_status, model_names = endpoint_results[(config.provider, endpoint, credential_id)]
        if endpoint_status != "reachable":
            return "unreachable"
        configured_model = adapter.configured_model_name(config)
        if configured_model not in model_names and f"{configured_model}:latest" not in model_names:
            return "model_unavailable"
        return "reachable"

    llm_status = model_status(settings.llm)
    guardrail_status = model_status(settings.guardrail_llm) if guardrail_enabled else "disabled"
    if not settings.conversation_history.enabled:
        history_status = "disabled"
    else:
        history_service = getattr(request.app.state, "conversation_history_service", None)
        try:
            history_status = "reachable" if history_service and await history_service.ping() else "unreachable"
        except Exception as exc:
            history_status = "unreachable"
            logger.warning("Redis readiness check failed error_type=%s", type(exc).__name__)
    if (
        llm_status != "reachable"
        or guardrail_status not in {"reachable", "disabled"}
        or history_status == "unreachable"
    ):
        logger.warning(
            "Dependency readiness check failed llm=%s guardrail=%s conversation_history=%s",
            llm_status,
            guardrail_status,
            history_status,
        )
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="unavailable",
            llm=llm_status,
            guardrail_llm=guardrail_status,
            conversation_history=history_status,
        )
    logger.debug("Dependency readiness check succeeded")
    return HealthResponse(
        status="ok",
        llm=llm_status,
        guardrail_llm=guardrail_status,
        conversation_history=history_status,
    )


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
