"""Safe API error representations shared by the application."""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Do not include request content or exception details in the client response.
        logger.exception(
            "Unhandled request error request_id=%s",
            getattr(request.state, "request_id", "unknown"),
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred.",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )
