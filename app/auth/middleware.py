"""Small bootstrap authenticator; replace with the chosen identity provider later."""

from __future__ import annotations

import hmac
import logging
import os

from fastapi import HTTPException, Request, status

from app.auth.principal import Principal
from app.config import AppSettings


logger = logging.getLogger(__name__)


async def get_principal(request: Request) -> Principal:
    settings: AppSettings = request.app.state.settings
    if settings.auth.mode == "development":
        return Principal(
            user_id=request.headers.get("X-User-ID", settings.auth.development_user_id),
            household_id=request.headers.get(
                "X-Household-ID", settings.auth.development_household_id
            ),
            auth_method="development",
        )

    authorization = request.headers.get("Authorization", "")
    scheme, _, supplied_token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not supplied_token:
        logger.warning("Authentication rejected request_id=%s reason=missing_bearer", request.state.request_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid bearer token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    for credential in settings.auth.credentials:
        expected_token = os.getenv(credential.token_environment_variable, "")
        if expected_token and hmac.compare_digest(supplied_token, expected_token):
            logger.debug(
                "Bearer authentication succeeded request_id=%s user_id=%s",
                request.state.request_id,
                credential.user_id,
            )
            return Principal(
                user_id=credential.user_id,
                household_id=credential.household_id,
                auth_method="bearer",
            )

    logger.warning("Authentication rejected request_id=%s reason=invalid_bearer", request.state.request_id)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="A valid bearer token is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )
