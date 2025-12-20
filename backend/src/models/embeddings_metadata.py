"""
Embeddings metadata model for tracking indexed document chunks.

Stores metadata about each vector embedding in Qdrant for retrieval
and citation purposes.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Index, Integer, String, Text, UUID, func

from src.models.database import Base
import uuid


class EmbeddingsMetadata(Base):
    """
    Metadata for document chunks indexed in Qdrant.

    Attributes:
        id: Unique metadata ID (UUID)
        chunk_id: Identifier matching Qdrant point ID
        chapter: Chapter number or name from textbook
        section: Section title or number
        subsection: Subsection for deeper organization (optional)
        difficulty_level: Content difficulty (beginner, intermediate, advanced)
        token_count: Number of tokens in chunk
        source_url: URL or location of source document
        content_preview: First 200 chars of chunk content (optional)
        created_at: Timestamp when indexed
    """

    __tablename__ = "embeddings_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chunk_id = Column(String(255), nullable=False, unique=True, index=True)
    chapter = Column(String(255), nullable=False, index=True)
    section = Column(String(255), nullable=False, index=True)
    subsection = Column(String(255), nullable=True)
    difficulty_level = Column(String(50), default="intermediate", nullable=False)
    token_count = Column(Integer, nullable=False)
    source_url = Column(String(500), nullable=True)
    content_preview = Column(Text, nullable=True)  # First 200 chars
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Indexes for efficient querying
    __table_args__ = (
        Index("idx_chapter_section", "chapter", "section"),
        Index("idx_difficulty", "difficulty_level"),
        Index("idx_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """String representation of EmbeddingsMetadata."""
        return (
            f"<EmbeddingsMetadata(chunk_id={self.chunk_id}, "
            f"chapter={self.chapter}, section={self.section})>"
        )

    def to_dict(self) -> dict:
        """Convert metadata to dictionary."""
        return {
            "id": str(self.id),
            "chunk_id": self.chunk_id,
            "chapter": self.chapter,
            "section": self.section,
            "subsection": self.subsection,
            "difficulty_level": self.difficulty_level,
            "token_count": self.token_count,
            "source_url": self.source_url,
            "content_preview": self.content_preview,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
