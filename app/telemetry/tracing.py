"""OpenTelemetry tracing setup with an OTLP exporter compatible with Jaeger."""

from __future__ import annotations

import logging
from weakref import WeakSet

import httpx
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from app.config import TracingConfig


logger = logging.getLogger(__name__)
_provider: TracerProvider | None = None
_instrumented_apps: WeakSet[FastAPI] = WeakSet()


def configure_tracing(application: FastAPI, config: TracingConfig) -> TracerProvider | None:
    """Configure process-wide tracing and instrument this FastAPI application.

    OpenTelemetry's global provider is process-wide. This initializer is
    idempotent so application factories used by local tooling do not register
    multiple providers or duplicate instrumentation.
    """
    global _provider

    if not config.enabled:
        logger.info("OpenTelemetry tracing disabled")
        return None

    if _provider is None:
        resource = Resource.create({SERVICE_NAME: config.service_name})
        provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(config.sample_ratio),
        )
        exporter = OTLPSpanExporter(
            endpoint=str(config.otlp_endpoint),
            timeout=config.export_timeout_seconds,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _provider = provider
        logger.info(
            "OpenTelemetry tracing configured service=%s endpoint=%s sample_ratio=%s",
            config.service_name,
            config.otlp_endpoint,
            config.sample_ratio,
        )

    if application not in _instrumented_apps:
        FastAPIInstrumentor.instrument_app(application, tracer_provider=_provider)
        _instrumented_apps.add(application)

    return _provider


def instrument_http_client(client: httpx.AsyncClient, provider: TracerProvider | None) -> None:
    """Instrument the app's shared HTTPX client for the outgoing LLM request."""
    if provider is None:
        return
    HTTPXClientInstrumentor().instrument_client(client, tracer_provider=provider)


def flush_tracing(provider: TracerProvider | None) -> None:
    """Flush queued spans before the process or application stops."""
    if provider is not None:
        provider.force_flush(timeout_millis=5000)
