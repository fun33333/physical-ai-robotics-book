"""
Utility functions for RAG Chatbot backend.

Provides structured logging, error handling, and latency tracking utilities.
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator

from src.config import settings


def setup_logging() -> None:
    """Set up structured JSON logging for the application."""
    logging.basicConfig(
        level=settings.log_level,
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',  # pylint: disable=line-too-long
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


@contextmanager
def track_latency(operation_name: str) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager to track and log operation latency.

    Usage:
        with track_latency("rag_retrieval") as latency:
            # perform operation
            pass
        # latency["elapsed_ms"] contains the elapsed time

    Args:
        operation_name: Name of the operation being tracked

    Yields:
        Dictionary to store latency information
    """
    latency_info: Dict[str, Any] = {"operation": operation_name, "start_time": 0, "elapsed_ms": 0}
    start_time = time.time()
    latency_info["start_time"] = start_time

    try:
        yield latency_info
    finally:
        elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
        latency_info["elapsed_ms"] = round(elapsed, 2)
        logger = get_logger(__name__)
        logger.info(
            json.dumps({
                "operation": operation_name,
                "elapsed_ms": latency_info["elapsed_ms"],
                "type": "latency_tracking",
            })
        )


class AppError(Exception):
    """Base exception for application errors."""

    def __init__(self, message: str, error_code: str = "INTERNAL_ERROR", status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        return {
            "error": self.error_code,
            "message": self.message,
            "status_code": self.status_code,
        }


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR", 400)


class QdrantError(AppError):
    """Raised when Qdrant operations fail."""

    def __init__(self, message: str):
        super().__init__(message, "QDRANT_ERROR", 503)


class GeminiError(AppError):
    """Raised when Gemini API operations fail."""

    def __init__(self, message: str):
        super().__init__(message, "GEMINI_ERROR", 503)


class DatabaseError(AppError):
    """Raised when database operations fail."""

    def __init__(self, message: str):
        super().__init__(message, "DATABASE_ERROR", 503)


class HallucinationDetectedError(AppError):
    """Raised when a hallucination is detected by Safety Guardian."""

    def __init__(self, message: str):
        super().__init__(message, "HALLUCINATION_DETECTED", 422)


def truncate_string(text: str, max_length: int = 250) -> str:
    """
    Truncate a string to max_length characters, preserving word boundaries.

    Args:
        text: Text to truncate
        max_length: Maximum length in characters

    Returns:
        Truncated text with ellipsis if truncated
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(" ", 1)[0]
    return f"{truncated}..."


def format_latency_breakdown(latencies: Dict[str, float]) -> str:
    """
    Format latency breakdown for logging.

    Args:
        latencies: Dictionary with stage names and latency values in ms

    Returns:
        Formatted latency breakdown string
    """
    parts = []
    for stage, ms in latencies.items():
        parts.append(f"{stage}:{ms:.0f}ms")
    return " | ".join(parts)
