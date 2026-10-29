"""FastAPI application factory and local entry point."""

from __future__ import annotations

import re
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.router import router
from app.config import AppSettings, load_settings
from app.errors import register_exception_handlers
from app.planning.classifier import LLMIntentClassifier
from app.security.input_guardrails import InputGuardrails
from app.security.rate_limits import InMemoryRateLimiter
from app.telemetry.logging import configure_logging
from app.workflow.request_flow import RequestWorkflow


REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


def create_app(settings: AppSettings | None = None) -> FastAPI:
    configured_settings = settings or load_settings()
    configure_logging(configured_settings.application.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(configured_settings.llm.timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            application.state.http_client = client
            classifier = LLMIntentClassifier(
                client=client,
                config=configured_settings.llm,
                catalog=configured_settings.camera_catalog,
            )
            application.state.request_workflow = RequestWorkflow(
                guardrails=InputGuardrails(configured_settings.guardrails),
                classifier=classifier,
            )
            application.state.rate_limiter = InMemoryRateLimiter(
                configured_settings.api.max_requests_per_minute
            )
            yield

    application = FastAPI(
        title=configured_settings.application.name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = configured_settings

    @application.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied_id = request.headers.get("X-Request-ID", "")
        request_id = supplied_id if REQUEST_ID_PATTERN.fullmatch(supplied_id) else str(uuid.uuid4())
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500,
                content={
                    "detail": "An unexpected error occurred.",
                    "request_id": request_id,
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response

    application.include_router(router)
    register_exception_handlers(application)
    return application


app = create_app()
