"""Redis-backed, per-user conversation history for the request workflow."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.auth.principal import Principal


logger = logging.getLogger(__name__)
MAX_CONVERSATION_TURNS = 5


class ConversationTurn(BaseModel):
    """One completed user/assistant exchange."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    user_message: str = Field(min_length=1, max_length=2000)
    agent_message: str = Field(min_length=1, max_length=2000)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationHistoryService:
    """Read and atomically append a FIFO of at most five completed exchanges."""

    def __init__(self, client: Any, key_prefix: str, ttl_seconds: int) -> None:
        self._client = client
        self._key_prefix = key_prefix.rstrip(":")
        self._ttl_seconds = ttl_seconds

    @staticmethod
    def _identity_digest(principal: Principal) -> str:
        identity = f"{principal.household_id}\0{principal.user_id}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()

    def _key(self, principal: Principal) -> str:
        return f"{self._key_prefix}:{self._identity_digest(principal)}"

    async def get_recent_turns(self, principal: Principal) -> list[ConversationTurn]:
        values = await self._client.lrange(self._key(principal), 0, -1)
        turns: list[ConversationTurn] = []
        for value in values:
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            try:
                turns.append(ConversationTurn.model_validate_json(value))
            except (TypeError, ValueError):
                logger.warning("Invalid conversation history entry ignored")
        return turns[-MAX_CONVERSATION_TURNS:]

    async def append_turn(
        self,
        principal: Principal,
        user_message: str,
        agent_message: str,
    ) -> list[ConversationTurn]:
        turn = ConversationTurn(user_message=user_message, agent_message=agent_message)
        key = self._key(principal)
        async with self._client.pipeline(transaction=True) as pipeline:
            pipeline.rpush(key, turn.model_dump_json())
            pipeline.ltrim(key, -MAX_CONVERSATION_TURNS, -1)
            pipeline.expire(key, self._ttl_seconds)
            await pipeline.execute()
        return await self.get_recent_turns(principal)

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def clear_history(self, principal: Principal) -> None:
        await self._client.delete(self._key(principal))

    async def close(self) -> None:
        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


class UnavailableRedisClient:
    """Graceful fallback used only when the Redis dependency is not installed."""

    async def ping(self) -> bool:
        raise RuntimeError("Redis client dependency is unavailable")

    async def lrange(self, *_: object) -> list[bytes]:
        raise RuntimeError("Redis client dependency is unavailable")

    async def delete(self, *_: object) -> int:
        raise RuntimeError("Redis client dependency is unavailable")

    def pipeline(self, **_: object) -> Any:
        raise RuntimeError("Redis client dependency is unavailable")

    async def aclose(self) -> None:
        return None


def create_redis_client(url: str, connect_timeout_seconds: float, socket_timeout_seconds: float) -> Any:
    """Create an async Redis client without making Redis a module import-time dependency."""
    try:
        from redis.asyncio import Redis
    except ImportError:
        logger.error("Redis client dependency is unavailable; conversation history is disabled")
        return UnavailableRedisClient()
    return Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=connect_timeout_seconds,
        socket_timeout=socket_timeout_seconds,
        health_check_interval=30,
    )
