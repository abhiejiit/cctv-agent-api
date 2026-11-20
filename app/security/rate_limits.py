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
        self._max_tracked_users = 10_000
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, user_id: str) -> None:
        now = time.monotonic()
        cutoff = now - 60
        async with self._lock:
            if user_id not in self._events and len(self._events) >= self._max_tracked_users:
                for key in tuple(self._events):
                    events = self._events[key]
                    while events and events[0] <= cutoff:
                        events.popleft()
                    if not events:
                        self._events.pop(key, None)
                    if len(self._events) < self._max_tracked_users:
                        break
                if len(self._events) >= self._max_tracked_users:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Rate limiter capacity is temporarily exhausted.",
                        headers={"Retry-After": "1"},
                    )
            events = self._events[user_id]
            while events and events[0] <= cutoff:
                events.popleft()

            if len(events) >= self._limit:
                retry_after = max(1, int(60 - (now - events[0])))
                logger.warning("Rate limit exceeded retry_after=%s", retry_after)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Request rate limit exceeded.",
                    headers={"Retry-After": str(retry_after)},
                )

            events.append(now)
