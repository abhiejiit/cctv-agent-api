"""Python camera tools that delegate data retrieval to the camera service."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar, Token
from typing import Callable

from opentelemetry import trace
from opentelemetry.trace import Span
from opentelemetry.trace import Status, StatusCode

from app.orchestration.tool_schemas import (
    BackEntranceToolInput,
    CommonLawnToolInput,
    FrontEntranceToolInput,
    ParkingOccupancyToolInput,
    ParkingToolInput,
)
from app.services.camera_service_provider import get_camera_data_service

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)
_tool_trace_options: ContextVar[tuple[bool, bool]] = ContextVar(
    "camera_tool_trace_options", default=(True, True)
)
_intent_agent_span: ContextVar[Span | None] = ContextVar("camera_tool_agent_span", default=None)
_MAX_TOOL_TRACE_CHARS = 12_000


def bind_tool_trace_options(capture_inputs: bool, capture_outputs: bool) -> Token:
    """Set per-request policy for tool payload capture in OpenTelemetry spans."""
    return _tool_trace_options.set((capture_inputs, capture_outputs))


def reset_tool_trace_options(token: Token) -> None:
    _tool_trace_options.reset(token)


def bind_intent_agent_span(span: Span) -> Token:
    """Keep a direct reference to the intent-agent parent span across tool calls."""
    return _intent_agent_span.set(span)


def reset_intent_agent_span(token: Token) -> None:
    _intent_agent_span.reset(token)


def _trace_service_call(
    tool_name: str,
    parameters: object,
    service_call: Callable[[], object],
) -> str:
    """Run a camera service call and record its full bounded input/output trace."""
    capture_inputs, capture_outputs = _tool_trace_options.get()
    parent_span = _intent_agent_span.get() or trace.get_current_span()
    input_json = parameters.model_dump_json()  # type: ignore[attr-defined]
    started_at = time.perf_counter()

    with tracer.start_as_current_span(f"cctv.crewai.tool.{tool_name}") as span:
        span.set_attribute("crewai.tool.name", tool_name)
        span.set_attribute("crewai.tool.input.schema", type(parameters).__name__)
        span.set_attribute("crewai.tool.input.length", len(input_json))
        if capture_inputs:
            span.set_attribute("crewai.tool.input.json", input_json[:_MAX_TOOL_TRACE_CHARS])
        if capture_inputs:
            logger.info("CrewAI tool started tool=%s input=%s", tool_name, input_json)
        else:
            logger.info("CrewAI tool started tool=%s input_length=%d", tool_name, len(input_json))
        try:
            result = service_call()
            output_json = result.model_dump_json()  # type: ignore[attr-defined]
            elapsed_ms = round((time.perf_counter() - started_at) * 1000)
            span.set_attribute("crewai.tool.status", "success")
            span.set_attribute("crewai.tool.output.length", len(output_json))
            span.set_attribute("crewai.tool.execution.elapsed_ms", elapsed_ms)
            span.set_attribute("crewai.tool.camera_id", result.camera_id)  # type: ignore[attr-defined]
            span.set_attribute("crewai.tool.data_source", result.source)  # type: ignore[attr-defined]
            if capture_outputs:
                span.set_attribute(
                    "crewai.tool.output.json",
                    output_json[:_MAX_TOOL_TRACE_CHARS],
                )
                span.set_attribute(
                    "crewai.tool.output.truncated",
                    len(output_json) > _MAX_TOOL_TRACE_CHARS,
                )
            # Copy the latest result onto the agent span so it is visible without
            # opening the tool child span. The child span retains every call.
            if parent_span.is_recording():
                parent_span.set_attribute("crewai.tool.last.name", tool_name)
                parent_span.set_attribute("crewai.tool.last.status", "success")
                parent_span.set_attribute("crewai.tool.last.execution.elapsed_ms", elapsed_ms)
                if capture_inputs:
                    parent_span.set_attribute(
                        "crewai.tool.last.input.json",
                        input_json[:_MAX_TOOL_TRACE_CHARS],
                    )
                if capture_outputs:
                    parent_span.set_attribute(
                        "crewai.tool.last.output.json",
                        output_json[:_MAX_TOOL_TRACE_CHARS],
                    )
            logger.info(
                "CrewAI tool completed tool=%s elapsed_ms=%d output_length=%d output=%s",
                tool_name,
                elapsed_ms,
                len(output_json),
                output_json if capture_outputs else "<capture disabled>",
            )
            return output_json
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started_at) * 1000)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            span.set_attribute("crewai.tool.status", "error")
            span.set_attribute("crewai.tool.error.type", type(exc).__name__)
            span.set_attribute("crewai.tool.error.message", str(exc)[:1000])
            span.set_attribute("crewai.tool.execution.elapsed_ms", elapsed_ms)
            if parent_span.is_recording():
                parent_span.set_attribute("crewai.tool.last.name", tool_name)
                parent_span.set_attribute("crewai.tool.last.status", "error")
                parent_span.set_attribute("crewai.tool.last.error.type", type(exc).__name__)
                parent_span.set_attribute("crewai.tool.last.error.message", str(exc)[:1000])
            logger.exception("CrewAI tool failed tool=%s elapsed_ms=%d", tool_name, elapsed_ms)
            raise


def fetch_front_entrance_details(
    entity_requested: str,
    user_question: str,
    property_requested: str | None = None,
) -> str:
    """Call the camera service for an entity and optional property at the front entrance."""
    params = FrontEntranceToolInput(
        entity_requested=entity_requested,
        property_requested=property_requested,
        user_question=user_question,
    )
    return _trace_service_call(
        "fetch_front_entrance_details",
        params,
        lambda: get_camera_data_service().fetch_entity(
            camera_id="demo-front-entrance",
            entity_requested=params.entity_requested,
            property_requested=params.property_requested,
            user_question=params.user_question,
        ),
    )


def fetch_parking_details(
    entity_requested: str,
    user_question: str,
    property_requested: str | None = None,
) -> str:
    """Call the camera service for an entity and optional property in parking."""
    params = ParkingToolInput(
        entity_requested=entity_requested,
        property_requested=property_requested,
        user_question=user_question,
    )
    return _trace_service_call(
        "fetch_parking_details",
        params,
        lambda: get_camera_data_service().fetch_entity(
            camera_id="demo-parking",
            entity_requested=params.entity_requested,
            property_requested=params.property_requested,
            user_question=params.user_question,
        ),
    )


def get_parking_occupancy(metric_requested: str, user_question: str) -> str:
    """Call the camera service for requested parking occupancy statistics."""
    params = ParkingOccupancyToolInput(
        metric_requested=metric_requested,
        user_question=user_question,
    )
    return _trace_service_call(
        "get_parking_occupancy",
        params,
        lambda: get_camera_data_service().fetch_parking_summary(
            metric_requested=params.metric_requested,
            user_question=params.user_question,
        ),
    )


def fetch_back_entrance_details(
    entity_requested: str,
    user_question: str,
    property_requested: str | None = None,
) -> str:
    """Call the camera service for an entity and optional property at the back entrance."""
    params = BackEntranceToolInput(
        entity_requested=entity_requested,
        property_requested=property_requested,
        user_question=user_question,
    )
    return _trace_service_call(
        "fetch_back_entrance_details",
        params,
        lambda: get_camera_data_service().fetch_entity(
            camera_id="demo-back-entrance",
            entity_requested=params.entity_requested,
            property_requested=params.property_requested,
            user_question=params.user_question,
        ),
    )


def fetch_common_lawn_details(
    entity_requested: str,
    user_question: str,
    property_requested: str | None = None,
) -> str:
    """Call the camera service for an entity and optional property on the common lawn."""
    params = CommonLawnToolInput(
        entity_requested=entity_requested,
        property_requested=property_requested,
        user_question=user_question,
    )
    return _trace_service_call(
        "fetch_common_lawn_details",
        params,
        lambda: get_camera_data_service().fetch_entity(
            camera_id="demo-common-lawn",
            entity_requested=params.entity_requested,
            property_requested=params.property_requested,
            user_question=params.user_question,
        ),
    )


def _tool_registry() -> dict[str, object]:
    """Wrap Python functions as CrewAI tools with explicit Pydantic schemas."""
    from crewai.tools.base_tool import Tool

    return {
        "fetch_front_entrance_details": Tool(
            name="fetch_front_entrance_details",
            description="For front entrance questions. Extract entity_requested, optional property_requested, and the original user_question before fetching the camera snapshot.",
            func=fetch_front_entrance_details,
            args_schema=FrontEntranceToolInput,
        ),
        "fetch_parking_details": Tool(
            name="fetch_parking_details",
            description="For a specific object in parking. Extract entity_requested, optional property_requested, and the original user_question before fetching the camera snapshot.",
            func=fetch_parking_details,
            args_schema=ParkingToolInput,
        ),
        "get_parking_occupancy": Tool(
            name="get_parking_occupancy",
            description="For general parking availability, occupied spaces, or vehicle count. Provide metric_requested and user_question.",
            func=get_parking_occupancy,
            args_schema=ParkingOccupancyToolInput,
        ),
        "fetch_back_entrance_details": Tool(
            name="fetch_back_entrance_details",
            description="For back entrance questions. Extract entity_requested, optional property_requested, and the original user_question before fetching the camera snapshot.",
            func=fetch_back_entrance_details,
            args_schema=BackEntranceToolInput,
        ),
        "fetch_common_lawn_details": Tool(
            name="fetch_common_lawn_details",
            description="For common lawn questions. Extract entity_requested, optional property_requested, and the original user_question before fetching the camera snapshot.",
            func=fetch_common_lawn_details,
            args_schema=CommonLawnToolInput,
        ),
    }


def resolve_agent_tools(tool_names: list[str]) -> list[object]:
    """Resolve configured tool names against the static Python function registry."""
    registry = _tool_registry()
    unknown = sorted(set(tool_names) - registry.keys())
    if unknown:
        raise ValueError(f"Unknown configured CrewAI tools: {', '.join(unknown)}")
    return [registry[name] for name in tool_names]


AVAILABLE_TOOL_NAMES = frozenset(
    {
        "fetch_front_entrance_details",
        "fetch_parking_details",
        "get_parking_occupancy",
        "fetch_back_entrance_details",
        "fetch_common_lawn_details",
    }
)
