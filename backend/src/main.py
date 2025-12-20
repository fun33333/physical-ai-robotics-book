"""
FastAPI Application for RAG Chatbot Backend.

Entry point for the multi-agent RAG chatbot service. Configures FastAPI app,
middleware, and route handlers.
"""

import time
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import settings
from src.utils import get_logger, setup_logging

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for textbook Q&A with orchestrator agent system",
    version="0.1.0",
)

# Include API routes
app.include_router(router)

# CORS Middleware
cors_origins = [origin.strip() for origin in settings.cors_origin.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate Limiting Middleware (simplified version)
class RateLimitMiddleware:
    """Simple rate limiting middleware (15 req/min per IP)."""

    def __init__(self, app: FastAPI, requests_per_minute: int = 15):
        self.app = app
        self.requests_per_minute = requests_per_minute
        self.client_requests: Dict[str, list] = {}

    async def __call__(self, request: Request, call_next: Any) -> Any:
        """Process request with rate limiting."""
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Initialize client request list if not exists
        if client_ip not in self.client_requests:
            self.client_requests[client_ip] = []

        # Remove requests older than 1 minute
        self.client_requests[client_ip] = [
            req_time for req_time in self.client_requests[client_ip]
            if now - req_time < 60
        ]

        # Check if rate limit exceeded
        if len(self.client_requests[client_ip]) >= self.requests_per_minute:
            return {
                "error": "RATE_LIMIT_EXCEEDED",
                "message": f"Rate limit exceeded: {self.requests_per_minute} requests per minute",
                "status_code": 429,
            }

        # Add current request to tracking
        self.client_requests[client_ip].append(now)

        response = await call_next(request)
        return response


# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)


# Request/Response Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log incoming requests and responses."""
    request_id = f"{time.time()}"

    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000

    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"Status: {response.status_code} | "
        f"Duration: {process_time:.2f}ms | "
        f"Request ID: {request_id}"
    )

    return response


# Error Handlers
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> Dict[str, Any]:
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": "INTERNAL_ERROR",
        "message": "An unexpected error occurred",
        "status_code": 500,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
    )
