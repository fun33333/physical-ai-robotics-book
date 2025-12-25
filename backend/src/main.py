"""
FastAPI Application for RAG Chatbot Backend.
"""

import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router
from src.api.middleware import RateLimitMiddleware, InputSanitizationMiddleware, get_cors_origins
from src.config import settings
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Multi-Agent RAG Chatbot API",
    description="RAG chatbot with OpenAI Agents SDK + Gemini",
    version="0.1.0",
)

# Routes
app.include_router(router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate Limiting
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)

# Input Sanitization
app.add_middleware(InputSanitizationMiddleware)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log requests with timing."""
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration:.0f}ms)")
    return response


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)
