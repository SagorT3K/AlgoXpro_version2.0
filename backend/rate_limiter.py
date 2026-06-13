"""Per-IP rate limiting middleware for FastAPI.

Limits:
- /api/analyze: 10 requests per minute per IP
- /api/assets-live: 30 requests per minute per IP
- WebSocket connections: 3 per IP
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self):
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._cleanup_task: asyncio.Task | None = None

    def _cleanup(self):
        """Remove old entries every 60s."""
        now = time.time()
        expired = [k for k, v in self._buckets.items() if not v or now - v[-1] > 120]
        for k in expired:
            del self._buckets[k]

    def is_allowed(self, key: str, limit: int, window: float = 60.0) -> bool:
        """Check if request is allowed under the rate limit."""
        now = time.time()
        # Remove expired entries
        self._buckets[key] = [t for t in self._buckets[key] if now - t < window]
        if len(self._buckets[key]) >= limit:
            return False
        self._buckets[key].append(now)
        return True

    def get_remaining(self, key: str, limit: int, window: float = 60.0) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        self._buckets[key] = [t for t in self._buckets[key] if now - t < window]
        return max(0, limit - len(self._buckets[key]))


_rate_limiter = RateLimiter()

# Rate limits per endpoint pattern
RATE_LIMITS = {
    "/api/analyze": (10, 60),      # 10 per minute
    "/api/assets-live": (30, 60),  # 30 per minute
    "/api/assets": (20, 60),       # 20 per minute
    "/api/health": (60, 60),       # 60 per minute
    "/api/reconnect": (5, 60),     # 5 per minute
}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that enforces per-IP rate limits."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for static files and frontend
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        # Get client IP (handle proxies)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "")
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"

        # Find matching rate limit
        for pattern, (limit, window) in RATE_LIMITS.items():
            if path.startswith(pattern):
                key = f"{client_ip}:{pattern}"
                if not _rate_limiter.is_allowed(key, limit, window):
                    remaining = _rate_limiter.get_remaining(key, limit, window)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "retry_after": int(window),
                            "remaining": remaining,
                        },
                        headers={
                            "X-RateLimit-Limit": str(limit),
                            "X-RateLimit-Remaining": "0",
                            "Retry-After": str(int(window)),
                        },
                    )
                # Add rate limit headers
                remaining = _rate_limiter.get_remaining(key, limit, window)
                response = await call_next(request)
                response.headers["X-RateLimit-Limit"] = str(limit)
                response.headers["X-RateLimit-Remaining"] = str(remaining)
                return response

        return await call_next(request)


# WebSocket connection tracking
_ws_connections: dict[str, int] = defaultdict(int)
MAX_WS_PER_IP = 3


def can_open_ws(ip: str) -> bool:
    """Check if IP can open another WebSocket connection."""
    return _ws_connections[ip] < MAX_WS_PER_IP


def register_ws(ip: str):
    """Register a new WebSocket connection."""
    _ws_connections[ip] += 1


def unregister_ws(ip: str):
    """Unregister a WebSocket connection."""
    _ws_connections[ip] = max(0, _ws_connections[ip] - 1)
    if _ws_connections[ip] == 0:
        del _ws_connections[ip]
