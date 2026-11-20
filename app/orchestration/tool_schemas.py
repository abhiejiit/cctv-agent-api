"""Pydantic input contracts exposed to CrewAI camera tools."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CameraToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    entity_requested: str = Field(
        min_length=1,
        max_length=40,
        description="Object class inferred from the user's question, such as door, person, or car.",
    )
    property_requested: str | None = Field(
        default=None,
        max_length=40,
        description="Requested attribute such as presence, count, or open_closed; omit when not applicable.",
    )
    user_question: str = Field(
        min_length=1,
        max_length=500,
        description="Short original camera question used to interpret the requested entity and property.",
    )


class FrontEntranceToolInput(CameraToolInput):
    pass


class ParkingToolInput(CameraToolInput):
    pass


class BackEntranceToolInput(CameraToolInput):
    pass


class CommonLawnToolInput(CameraToolInput):
    pass


class ParkingOccupancyToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    metric_requested: Literal["available_spaces", "occupied_spaces", "vehicle_count"] = Field(
        description="Parking statistic requested by the user."
    )
    user_question: str = Field(
        min_length=1,
        max_length=500,
        description="Short original parking question for context.",
    )
