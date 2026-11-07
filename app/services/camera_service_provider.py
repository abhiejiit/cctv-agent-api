"""Camera service selection point used by Python tools."""

from app.services.camera_data_service import CameraDataService
from app.services.mock_camera_data_service import MockCameraDataService

# Replace this provider with a configured live implementation when the camera
# analysis service is available. Agent tools keep using the same contract.
_camera_data_service: CameraDataService = MockCameraDataService()


def get_camera_data_service() -> CameraDataService:
    """Return the configured camera data-service adapter."""
    return _camera_data_service
