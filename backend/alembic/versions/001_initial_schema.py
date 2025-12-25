"""Initial schema for RAG Chatbot.

Creates tables for:
- conversations: User chat history and context
- embeddings_metadata: Metadata for indexed document chunks
- api_key_quota: API key usage tracking for rotation

Revision ID: 001_initial
Revises:
Create Date: 2025-12-21

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("selected_text", sa.Text(), nullable=True),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("agent_used", sa.String(100), nullable=False),
        sa.Column("tone", sa.String(50), nullable=False, server_default="english"),
        sa.Column("user_level", sa.String(50), nullable=False, server_default="intermediate"),
        sa.Column("sources", postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for conversations
    op.create_index("idx_conversations_user_id", "conversations", ["user_id"])
    op.create_index("idx_conversations_session_id", "conversations", ["session_id"])
    op.create_index("idx_user_session", "conversations", ["user_id", "session_id"])
    op.create_index("idx_user_created", "conversations", ["user_id", "created_at"])

    # Create embeddings_metadata table
    op.create_table(
        "embeddings_metadata",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("chunk_id", sa.String(255), nullable=False),
        sa.Column("chapter", sa.String(255), nullable=False),
        sa.Column("section", sa.String(255), nullable=False),
        sa.Column("subsection", sa.String(255), nullable=True),
        sa.Column("difficulty_level", sa.String(50), nullable=False, server_default="intermediate"),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("source_url", sa.String(500), nullable=True),
        sa.Column("content_preview", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("chunk_id"),
    )

    # Create indexes for embeddings_metadata
    op.create_index("idx_embeddings_chunk_id", "embeddings_metadata", ["chunk_id"])
    op.create_index("idx_embeddings_chapter", "embeddings_metadata", ["chapter"])
    op.create_index("idx_embeddings_section", "embeddings_metadata", ["section"])
    op.create_index("idx_chapter_section", "embeddings_metadata", ["chapter", "section"])
    op.create_index("idx_difficulty", "embeddings_metadata", ["difficulty_level"])
    op.create_index("idx_embeddings_created_at", "embeddings_metadata", ["created_at"])

    # Create api_key_quota table
    op.create_table(
        "api_key_quota",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("api_key_id", sa.String(50), nullable=False),
        sa.Column("requests_today", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("requests_per_minute_today", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_reset", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("status", sa.String(50), nullable=False, server_default="active"),
        sa.Column("last_rotated_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("api_key_id"),
    )

    # Create index for api_key_quota
    op.create_index("idx_api_key_id", "api_key_quota", ["api_key_id"])


def downgrade() -> None:
    """Drop all tables."""
    # Drop api_key_quota
    op.drop_index("idx_api_key_id", table_name="api_key_quota")
    op.drop_table("api_key_quota")

    # Drop embeddings_metadata
    op.drop_index("idx_embeddings_created_at", table_name="embeddings_metadata")
    op.drop_index("idx_difficulty", table_name="embeddings_metadata")
    op.drop_index("idx_chapter_section", table_name="embeddings_metadata")
    op.drop_index("idx_embeddings_section", table_name="embeddings_metadata")
    op.drop_index("idx_embeddings_chapter", table_name="embeddings_metadata")
    op.drop_index("idx_embeddings_chunk_id", table_name="embeddings_metadata")
    op.drop_table("embeddings_metadata")

    # Drop conversations
    op.drop_index("idx_user_created", table_name="conversations")
    op.drop_index("idx_user_session", table_name="conversations")
    op.drop_index("idx_conversations_session_id", table_name="conversations")
    op.drop_index("idx_conversations_user_id", table_name="conversations")
    op.drop_table("conversations")
