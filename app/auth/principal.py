"""Authenticated caller identity carried through a request."""

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1, max_length=128)
    household_id: str = Field(min_length=1, max_length=128)
    auth_method: str = Field(min_length=1, max_length=32)
