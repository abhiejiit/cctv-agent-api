"""Application-scoped dependency accessors for route modules."""

from fastapi import Request

from app.config import AppSettings


def get_settings(request: Request) -> AppSettings:
    return request.app.state.settings
