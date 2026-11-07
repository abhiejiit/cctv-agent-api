"""Mock camera functions exposed as configurable CrewAI tools.

The registry is deliberately static: YAML can enable or select known tools, but
cannot import arbitrary Python code or execute an arbitrary function.
"""

from __future__ import annotations

import json
import logging
from typing import Literal


logger = logging.getLogger(__name__)
EntranceEntity = Literal["door", "window", "person", "vehicle", "package", "chair"]
ParkingEntity = Literal["car", "vehicle", "person", "bicycle", "motorcycle"]
LawnEntity = Literal["person", "dog", "cat", "bird", "vehicle", "chair"]


def _camera_payload(camera_id: str, entity: str) -> str:
    """Return representative mock detections for one camera and entity."""
    payload = {
        "source": "mock",
        "camera_id": camera_id,
        "entity": entity,
        "detected": False,
        "count": 0,
        "observations": [],
    }
    logger.info(
        "Camera mock tool returned camera_id=%s entity=%s detected=%s count=%d",
        camera_id,
        entity,
        payload["detected"],
        payload["count"],
    )
    return json.dumps(payload)


def _parking_occupancy() -> str:
    payload = {
        "source": "mock",
        "camera_id": "camera-parking",
        "occupied_spaces": 0,
        "total_spaces": 8,
        "vehicles_detected": 0,
    }
    logger.info("Parking mock tool returned occupied_spaces=0 total_spaces=8")
    return json.dumps(payload)


def _tool_registry() -> dict[str, object]:
    """Create CrewAI tool wrappers lazily, so API config can load without CrewAI."""
    from crewai.tools import tool

    # CrewAI's decorator creates a Pydantic schema from function annotations.
    # Publish these aliases to module globals for Pydantic's forward-ref resolver.
    globals().update(
        EntranceEntity=EntranceEntity,
        ParkingEntity=ParkingEntity,
        LawnEntity=LawnEntity,
    )

    @tool("fetch_front_entrance_details")
    def fetch_front_entrance_details(
        entity: EntranceEntity,
    ) -> str:
        """Fetch mock detections for an entity at the front entrance camera."""
        return _camera_payload("camera-front-entrance", entity)

    @tool("fetch_parking_details")
    def fetch_parking_details(
        entity: ParkingEntity,
    ) -> str:
        """Fetch mock detections for an entity at the parking camera."""
        return _camera_payload("camera-parking", entity)

    @tool("get_parking_occupancy")
    def get_parking_occupancy() -> str:
        """Fetch mock parking space occupancy for the parking camera."""
        return _parking_occupancy()

    @tool("fetch_back_entrance_details")
    def fetch_back_entrance_details(
        entity: EntranceEntity,
    ) -> str:
        """Fetch mock detections for an entity at the back entrance camera."""
        return _camera_payload("camera-back-entrance", entity)

    @tool("fetch_common_lawn_details")
    def fetch_common_lawn_details(
        entity: LawnEntity,
    ) -> str:
        """Fetch mock detections for an entity at the common lawn camera."""
        return _camera_payload("camera-common-lawn", entity)

    for wrapped in (
        fetch_front_entrance_details,
        fetch_parking_details,
        fetch_back_entrance_details,
        fetch_common_lawn_details,
    ):
        wrapped.args_schema.model_rebuild(
            _types_namespace={
                "EntranceEntity": EntranceEntity,
                "ParkingEntity": ParkingEntity,
                "LawnEntity": LawnEntity,
            }
        )

    return {
        "fetch_front_entrance_details": fetch_front_entrance_details,
        "fetch_parking_details": fetch_parking_details,
        "get_parking_occupancy": get_parking_occupancy,
        "fetch_back_entrance_details": fetch_back_entrance_details,
        "fetch_common_lawn_details": fetch_common_lawn_details,
    }


def resolve_agent_tools(tool_names: list[str]) -> list[object]:
    """Resolve configured tool names against the fixed, reviewed registry."""
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
