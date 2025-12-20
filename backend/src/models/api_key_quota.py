"""
API Key Quota tracking model.

Monitors usage of Gemini API keys for rotation and quota management
per the 15 requests/minute constraint.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, UUID, func

from src.models.database import Base
import uuid


class APIKeyQuota(Base):
    """
    Tracks API key usage and quotas.

    Attributes:
        id: Unique quota tracking ID (UUID)
        api_key_id: Identifier for the API key (e.g., "gemini_1", "gemini_2")
        requests_today: Number of requests made today
        last_reset: Timestamp of last quota reset (daily at 00:00 UTC)
        status: Current key status (active, exhausted, error)
        last_rotated_at: Timestamp of last rotation to next key
        error_message: Last error message if status is "error" (optional)
        created_at: Timestamp when tracking started
        updated_at: Timestamp of last update
    """

    __tablename__ = "api_key_quota"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    api_key_id = Column(String(50), nullable=False, unique=True, index=True)
    requests_today = Column(Integer, default=0, nullable=False)
    requests_per_minute_today = Column(Integer, default=0, nullable=False)
    last_reset = Column(DateTime, default=func.now(), nullable=False)
    status = Column(String(50), default="active", nullable=False)  # active, exhausted, error
    last_rotated_at = Column(DateTime, nullable=True)
    error_message = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        """String representation of APIKeyQuota."""
        return (
            f"<APIKeyQuota(api_key_id={self.api_key_id}, "
            f"requests_today={self.requests_today}, status={self.status})>"
        )

    def to_dict(self) -> dict:
        """Convert quota tracking to dictionary."""
        return {
            "id": str(self.id),
            "api_key_id": self.api_key_id,
            "requests_today": self.requests_today,
            "requests_per_minute_today": self.requests_per_minute_today,
            "last_reset": self.last_reset.isoformat() if self.last_reset else None,
            "status": self.status,
            "last_rotated_at": self.last_rotated_at.isoformat() if self.last_rotated_at else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def is_quota_exceeded(self, limit: int = 900) -> bool:
        """
        Check if daily request quota is exceeded.

        With 15 req/min limit, max per day = 15 * 60 * 24 = 21,600 requests
        Using 900 as soft limit for safety margin (15 req/min * 60 min).

        Args:
            limit: Request limit threshold

        Returns:
            True if requests_today exceeds limit
        """
        return self.requests_today >= limit

    def can_make_request(self, rpm_limit: int = 15) -> bool:
        """
        Check if a request can be made within rate limits.

        Args:
            rpm_limit: Requests per minute limit

        Returns:
            True if request can be made, False if rate limited
        """
        return self.status == "active" and self.requests_per_minute_today < rpm_limit
