"""Single-process sliding-window request limiter for the initial deployment."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque

from fastapi import HTTPException, status


logger = logging.getLogger(__name__)


class InMemoryRateLimiter:
    """Per-user limit; use a shared store before running multiple API replicas."""

    def __init__(self, requests_per_minute: int) -> None:
        self._limit = requests_per_minute
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, user_id: str) -> None:
        now = time.monotonic()
        cutoff = now - 60
        async with self._lock:
            events = self._events[user_id]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= self._limit:
                retry_after = max(1, int(60 - (now - events[0])))
                logger.warning("Rate limit exceeded user_id=%s retry_after=%s", user_id, retry_after)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Request rate limit exceeded.",
                    headers={"Retry-After": str(retry_after)},
                )

            events.append(now)
            # Discard idle keys so this local limiter does not grow indefinitely.
            if len(self._events) > 1024:
                for key in tuple(self._events):
                    if not self._events[key] or self._events[key][-1] <= cutoff:
                        self._events.pop(key, None)
