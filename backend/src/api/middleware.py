"""
FastAPI Middleware for CORS, Rate Limiting, and Input Sanitization.
"""

import time
from collections import defaultdict
from typing import Callable

from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)

# Rate limiting storage: {ip: [(timestamp, count)]}
_rate_limits: dict = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware (15 req/min per IP)."""

    def __init__(self, app, requests_per_minute: int = 15):
        super().__init__(app)
        self.rpm = requests_per_minute
        self.window = 60  # seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        _rate_limits[client_ip] = [
            t for t in _rate_limits[client_ip] if now - t < self.window
        ]

        # Check limit
        if len(_rate_limits[client_ip]) >= self.rpm:
            logger.warning(f"Rate limit exceeded: {client_ip}")
            return Response(
                content='{"error": "Rate limit exceeded. Try again later."}',
                status_code=429,
                headers={"X-RateLimit-Limit": str(self.rpm), "X-RateLimit-Remaining": "0"},
            )

        # Record request
        _rate_limits[client_ip].append(now)
        remaining = self.rpm - len(_rate_limits[client_ip])

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rpm)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


class InputSanitizationMiddleware(BaseHTTPMiddleware):
    """Sanitize and validate input."""

    MAX_BODY_SIZE = 50000  # 50KB

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.MAX_BODY_SIZE:
            return Response(
                content='{"error": "Request body too large"}',
                status_code=413,
            )
        return await call_next(request)


def get_cors_origins() -> list:
    """Get allowed CORS origins."""
    origins = [settings.cors_origin]
    if settings.environment == "development":
        origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])
    return origins
