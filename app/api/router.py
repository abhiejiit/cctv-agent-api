"""Root HTTP router."""

from fastapi import APIRouter

from app.api.routes.cctv_agent import router as cctv_agent_router
from app.api.routes.health import router as health_router


router = APIRouter()
router.include_router(cctv_agent_router)
router.include_router(health_router)
