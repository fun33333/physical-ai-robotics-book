"""
Conversation model for storing chat history.

Tracks user queries, responses, selected text context, and metadata
for multi-turn conversation support.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, String, Text, UUID, func, Index
from sqlalchemy.dialects.postgresql import JSON

from src.models.database import Base
import uuid


class Conversation(Base):
    """
    Stores conversation messages and metadata.

    Attributes:
        id: Unique conversation ID (UUID)
        user_id: User identifier (string)
        session_id: Session identifier for session isolation
        query: User's question or query
        selected_text: Text selected by user from textbook (optional)
        response: AI-generated response
        agent_used: Name of agent that generated response
        tone: Response tone (english, roman_urdu, bro_guide)
        user_level: User expertise level (beginner, intermediate, advanced)
        sources: JSON array of source citations
        created_at: Timestamp when created
        updated_at: Timestamp when last updated
    """

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    query = Column(Text, nullable=False)
    selected_text = Column(Text, nullable=True)
    response = Column(Text, nullable=False)
    agent_used = Column(String(100), nullable=False)  # e.g., "orchestrator", "rag_agent"
    tone = Column(String(50), default="english", nullable=False)
    user_level = Column(String(50), default="intermediate", nullable=False)
    sources = Column(JSON, nullable=True)  # List of {chapter, section, relevance_score}
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

    # Index for efficient querying by user and session
    __table_args__ = (
        Index("idx_user_session", "user_id", "session_id"),
        Index("idx_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of Conversation."""
        return (
            f"<Conversation(id={self.id}, user_id={self.user_id}, "
            f"tone={self.tone}, created_at={self.created_at})>"
        )

    def to_dict(self) -> dict:
        """Convert conversation to dictionary."""
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query": self.query,
            "selected_text": self.selected_text,
            "response": self.response,
            "agent_used": self.agent_used,
            "tone": self.tone,
            "user_level": self.user_level,
            "sources": self.sources,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
