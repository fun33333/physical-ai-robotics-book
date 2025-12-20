"""
Conversation History Service.

Manages conversation history storage, retrieval, and cleanup with
session isolation for multi-turn conversations.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from src.models.conversation import Conversation
from src.utils import DatabaseError, get_logger

logger = get_logger(__name__)


class ConversationService:
    """Service for managing conversation history and context."""

    def __init__(self, db_session: Session):
        """
        Initialize conversation service.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session

    def save_conversation(
        self,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        agent_used: str = "orchestrator",
        tone: str = "english",
        user_level: str = "intermediate",
        selected_text: Optional[str] = None,
        sources: Optional[List[dict]] = None,
    ) -> Conversation:
        """
        Save a conversation message to database.

        Args:
            user_id: User identifier
            session_id: Session identifier (for isolation)
            query: User's question
            response: AI-generated response
            agent_used: Name of agent that generated response
            tone: Response tone
            user_level: User expertise level
            selected_text: User-selected text context (optional)
            sources: List of source citations (optional)

        Returns:
            Saved Conversation object

        Raises:
            DatabaseError: If save fails
        """
        try:
            conversation = Conversation(
                user_id=user_id,
                session_id=session_id,
                query=query,
                response=response,
                agent_used=agent_used,
                tone=tone,
                user_level=user_level,
                selected_text=selected_text,
                sources=sources,
            )

            self.db_session.add(conversation)
            self.db_session.commit()
            self.db_session.refresh(conversation)

            logger.info(
                f"Saved conversation: user={user_id}, session={session_id}, "
                f"query_length={len(query)}"
            )
            return conversation

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to save conversation: {str(e)}")
            raise DatabaseError(f"Failed to save conversation: {str(e)}")

    def get_conversation_history(
        self,
        user_id: str,
        session_id: str,
        limit: int = 50,
    ) -> List[Conversation]:
        """
        Retrieve conversation history for a user session.

        Returns messages in chronological order (oldest first).

        Args:
            user_id: User identifier
            session_id: Session identifier
            limit: Maximum number of messages to return

        Returns:
            List of Conversation objects

        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            conversations = (
                self.db_session.query(Conversation)
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(Conversation.created_at)
                .limit(limit)
                .all()
            )

            logger.debug(f"Retrieved {len(conversations)} messages for user {user_id}")
            return conversations

        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {str(e)}")
            raise DatabaseError(f"Failed to retrieve history: {str(e)}")

    def get_recent_context(
        self,
        user_id: str,
        session_id: str,
        num_messages: int = 5,
    ) -> List[dict]:
        """
        Get recent conversation context for multi-turn responses.

        Returns the last N messages as formatted dictionaries for
        passing to agents.

        Args:
            user_id: User identifier
            session_id: Session identifier
            num_messages: Number of recent messages to include

        Returns:
            List of dictionaries with query and response
        """
        try:
            conversations = (
                self.db_session.query(Conversation)
                .filter_by(user_id=user_id, session_id=session_id)
                .order_by(desc(Conversation.created_at))
                .limit(num_messages)
                .all()
            )

            # Reverse to chronological order
            conversations = list(reversed(conversations))

            context = [
                {
                    "query": c.query,
                    "response": c.response,
                    "tone": c.tone,
                    "created_at": c.created_at.isoformat(),
                }
                for c in conversations
            ]

            return context

        except Exception as e:
            logger.warning(f"Failed to retrieve context: {str(e)}")
            return []

    def delete_old_conversations(self, older_than_days: int = 30) -> int:
        """
        Delete conversations older than specified days.

        Per constitution, deletes conversations > 30 days old to manage
        Neon 0.5GB free tier storage limit.

        Args:
            older_than_days: Delete messages older than this many days

        Returns:
            Number of conversations deleted

        Raises:
            DatabaseError: If deletion fails
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

            deleted_count = (
                self.db_session.query(Conversation)
                .filter(Conversation.created_at < cutoff_date)
                .delete()
            )

            self.db_session.commit()

            logger.info(f"Deleted {deleted_count} conversations older than {older_than_days} days")
            return deleted_count

        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Failed to delete old conversations: {str(e)}")
            raise DatabaseError(f"Failed to delete conversations: {str(e)}")

    def get_session_statistics(self, user_id: str, session_id: str) -> dict:
        """
        Get statistics for a user session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        try:
            conversations = (
                self.db_session.query(Conversation)
                .filter_by(user_id=user_id, session_id=session_id)
                .all()
            )

            tone_counts = {}
            for c in conversations:
                tone_counts[c.tone] = tone_counts.get(c.tone, 0) + 1

            return {
                "message_count": len(conversations),
                "tone_distribution": tone_counts,
                "session_duration": (
                    (conversations[-1].created_at - conversations[0].created_at).total_seconds()
                    if conversations
                    else 0
                ),
            }

        except Exception as e:
            logger.warning(f"Failed to get session statistics: {str(e)}")
            return {}
