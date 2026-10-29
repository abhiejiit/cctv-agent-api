"""Typed application configuration loaded from YAML with environment overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from app.planning.models import IntentLabel


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ApplicationConfig(StrictModel):
    name: str = "CCTV Finder Agent"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class ServerConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    request_timeout_seconds: float = Field(default=45, gt=0, le=300)


class ApiConfig(StrictModel):
    max_message_characters: int = Field(default=2000, ge=1, le=8192)
    max_requests_per_minute: int = Field(default=30, ge=1, le=10000)


class BearerCredentialConfig(StrictModel):
    token_environment_variable: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=128)
    household_id: str = Field(min_length=1, max_length=128)


class AuthConfig(StrictModel):
    mode: Literal["development", "bearer"] = "development"
    development_user_id: str = "dev-user"
    development_household_id: str = "demo-household"
    credentials: list[BearerCredentialConfig] = Field(default_factory=list)


class GuardrailPattern(StrictModel):
    category: str = Field(min_length=1, max_length=64)
    pattern: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_regex(self) -> "GuardrailPattern":
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"Invalid guardrail pattern for {self.category}") from exc
        return self


class GuardrailConfig(StrictModel):
    max_message_characters: int = Field(default=2000, ge=1, le=8192)
    blocked_patterns: list[GuardrailPattern] = Field(default_factory=list)


class LLMConfig(StrictModel):
    """Generic LLM endpoint configuration with a selectable provider adapter."""

    provider: Literal["ollama"] = "ollama"
    base_url: HttpUrl = "http://localhost:11434"
    model: str = Field(default="llama3.2:3b", min_length=1, max_length=128)
    chat_path: str = "/api/chat"
    readiness_path: str = "/api/tags"
    timeout_seconds: float = Field(default=35, gt=0, le=300)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=32, ge=1, le=512)
    keep_alive: str = Field(default="5m", min_length=1, max_length=32)


class CameraIntentConfig(StrictModel):
    description: str = Field(min_length=1, max_length=500)
    aliases: list[str] = Field(default_factory=list)
    camera_id: str | None = None
    entities: list[str] = Field(default_factory=list)
    properties: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_camera_definition(self) -> "CameraIntentConfig":
        if self.camera_id is None:
            if self.entities or self.properties:
                raise ValueError("Non-camera intents cannot define camera entities or properties")
            return self

        entity_names = set(self.entities)
        if set(self.properties) - entity_names:
            raise ValueError("Property definitions must refer to configured entities")
        return self


class CameraCatalog(StrictModel):
    intents: dict[IntentLabel, CameraIntentConfig]

    @model_validator(mode="after")
    def validate_intent_set(self) -> "CameraCatalog":
        expected = set(IntentLabel)
        configured = set(self.intents)
        if configured != expected:
            missing = sorted(intent.value for intent in expected - configured)
            extra = sorted(str(intent) for intent in configured - expected)
            raise ValueError(f"Camera catalog intent mismatch; missing={missing}, extra={extra}")
        return self


class AppSettings(StrictModel):
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    guardrails: GuardrailConfig = Field(default_factory=GuardrailConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    camera_catalog_file: str = "configs/cameras.yaml"
    camera_catalog: CameraCatalog | None = None

    @model_validator(mode="after")
    def validate_auth_mode(self) -> "AppSettings":
        if self.auth.mode == "development" and self.application.environment == "production":
            raise ValueError("Development header authentication is forbidden in production")
        if self.auth.mode == "bearer" and not self.auth.credentials:
            raise ValueError("Bearer authentication requires at least one configured credential")
        return self


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _apply_environment_overrides(raw: dict) -> dict:
    """Apply a small, explicit set of deployment overrides without hiding config."""
    data = {**raw}
    data["application"] = dict(data.get("application", {}))
    data["llm"] = dict(data.get("llm", {}))

    environment = os.getenv("CCTV_ENVIRONMENT")
    if environment:
        data["application"]["environment"] = environment

    llm_base_url = os.getenv("CCTV_LLM_BASE_URL")
    if llm_base_url:
        data["llm"]["base_url"] = llm_base_url

    llm_model = os.getenv("CCTV_LLM_MODEL")
    if llm_model:
        data["llm"]["model"] = llm_model

    return data


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load and validate application and camera configuration."""
    configured_path = config_path or os.getenv("CCTV_CONFIG_FILE", "configs/app.yaml")
    app_path = _resolve_path(configured_path)
    if not app_path.is_file():
        raise FileNotFoundError(f"Application config file not found: {app_path}")

    raw_config = yaml.safe_load(app_path.read_text(encoding="utf-8")) or {}
    settings = AppSettings.model_validate(_apply_environment_overrides(raw_config))

    cameras_path = _resolve_path(settings.camera_catalog_file)
    if not cameras_path.is_file():
        raise FileNotFoundError(f"Camera catalog file not found: {cameras_path}")
    camera_data = yaml.safe_load(cameras_path.read_text(encoding="utf-8")) or {}
    settings = AppSettings.model_validate(
        {**settings.model_dump(), "camera_catalog": CameraCatalog.model_validate(camera_data)}
    )

    if settings.auth.mode == "bearer":
        missing = [
            credential.token_environment_variable
            for credential in settings.auth.credentials
            if not os.getenv(credential.token_environment_variable)
        ]
        if missing:
            raise ValueError(f"Missing bearer token environment variables: {', '.join(missing)}")

    return settings
