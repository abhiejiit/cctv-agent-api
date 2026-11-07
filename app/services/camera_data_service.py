"""Camera data-service contract and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class Detection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[int] = Field(min_length=4, max_length=4, description="[x1, y1, x2, y2]")
    attributes: dict[str, str] = Field(default_factory=dict)


class CameraSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    camera_id: str
    captured_at: datetime
    requested_entity: str
    requested_property: str | None = None
    detected: bool
    count: int = Field(ge=0)
    detections: list[Detection] = Field(default_factory=list)


class ParkingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    camera_id: str
    captured_at: datetime
    total_spaces: int = Field(ge=0)
    occupied_spaces: int = Field(ge=0)
    available_spaces: int = Field(ge=0)
    vehicles_detected: int = Field(ge=0)
    metric_requested: str


class CameraDataService(Protocol):
    """Boundary that tools call to fetch the latest camera analysis result."""

    def fetch_entity(
        self,
        *,
        camera_id: str,
        entity_requested: str,
        property_requested: str | None,
        user_question: str,
    ) -> CameraSnapshot: ...

    def fetch_parking_summary(
        self,
        *,
        metric_requested: str,
        user_question: str,
    ) -> ParkingSummary: ...
