"""Temporary mock implementation of the camera data-service contract."""

from __future__ import annotations

from datetime import datetime, timezone

from app.services.camera_data_service import CameraSnapshot, CameraDataService, Detection, ParkingSummary


class MockCameraDataService(CameraDataService):
    """Return static YOLO-shaped detections until a live camera adapter is added."""

    _detections: dict[str, list[Detection]] = {
        "demo-front-entrance": [
            Detection(class_name="door", confidence=0.97, bbox=[284, 84, 518, 612], attributes={"open_closed": "closed"}),
            Detection(class_name="person", confidence=0.91, bbox=[112, 198, 202, 488], attributes={"position": "near entrance"}),
        ],
        "demo-parking": [
            Detection(class_name="car", confidence=0.96, bbox=[194, 166, 442, 379], attributes={"parking_space": "P1"}),
        ],
        "demo-back-entrance": [
            Detection(class_name="door", confidence=0.94, bbox=[331, 96, 526, 607], attributes={"open_closed": "closed"}),
        ],
        "demo-common-lawn": [
            Detection(class_name="person", confidence=0.89, bbox=[370, 218, 432, 396], attributes={"position": "center of lawn"}),
            Detection(class_name="animal", confidence=0.86, bbox=[447, 305, 501, 365], attributes={"kind": "dog", "position": "near the path"}),
        ],
    }

    def fetch_entity(
        self,
        *,
        camera_id: str,
        entity_requested: str,
        property_requested: str | None,
        user_question: str,
    ) -> CameraSnapshot:
        """Return detections filtered by the entity extracted by the agent."""
        del user_question  # Reserved for a real service's query/audit context.
        matches = [
            detection for detection in self._detections.get(camera_id, [])
            if detection.class_name.casefold() == entity_requested.casefold()
            or (
                entity_requested.casefold() == "vehicle"
                and detection.class_name in {"car", "truck", "bus", "motorcycle"}
            )
            or (
                detection.class_name == "animal"
                and detection.attributes.get("kind", "").casefold() == entity_requested.casefold()
            )
        ]
        return CameraSnapshot(
            source="mock_yolo_camera_service",
            camera_id=camera_id,
            captured_at=datetime.now(timezone.utc),
            requested_entity=entity_requested,
            requested_property=property_requested,
            detected=bool(matches),
            count=len(matches),
            detections=matches,
        )

    def fetch_parking_summary(
        self,
        *,
        metric_requested: str,
        user_question: str,
    ) -> ParkingSummary:
        """Return mock parking occupancy and the requested aggregate metric."""
        del user_question
        vehicles = sum(
            item.class_name in {"car", "truck", "bus", "motorcycle"}
            for item in self._detections["demo-parking"]
        )
        total_spaces = 8
        return ParkingSummary(
            source="mock_yolo_camera_service",
            camera_id="demo-parking",
            captured_at=datetime.now(timezone.utc),
            total_spaces=total_spaces,
            occupied_spaces=vehicles,
            available_spaces=max(total_spaces - vehicles, 0),
            vehicles_detected=vehicles,
            metric_requested=metric_requested,
        )
