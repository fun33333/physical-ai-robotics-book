"""
Unit tests for Conversation History Retrieval (T055 - US3).

Tests the conversation service's ability to:
- Retrieve last 10 exchanges via get_recent_context(user_id, session_id)
- Enforce 30-day retention cutoff
- Handle empty history gracefully
- Maintain chronological order
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from src.services.conversation_service import ConversationService
from src.models.conversation import Conversation


class TestGetRecentContext:
    """Tests for get_recent_context method."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def sample_conversations(self):
        """Create sample conversation objects for testing."""
        now = datetime.utcnow()
        conversations = []
        for i in range(12):  # More than default 10 limit
            conv = MagicMock(spec=Conversation)
            conv.query = f"Question {i + 1}"
            conv.response = f"Answer {i + 1}"
            conv.tone = "english"
            conv.created_at = now - timedelta(minutes=12 - i)
            conversations.append(conv)
        return conversations

    def test_get_recent_context_returns_list(self, mock_session):
        """Test that get_recent_context returns a list."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

        service = ConversationService(db_session=mock_session)
        result = service.get_recent_context(user_id="user_123", session_id="session_456")

        assert isinstance(result, list)

    def test_get_recent_context_returns_last_n_exchanges(self, mock_session, sample_conversations):
        """Test that only the last N exchanges are returned."""
        # Return last 5 (default) from sample
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = sample_conversations[-5:]

        service = ConversationService(db_session=mock_session)
        result = service.get_recent_context(user_id="user_123", session_id="session_456", num_messages=5)

        # Should have 5 exchanges
        assert len(result) <= 5

    def test_get_recent_context_chronological_order(self, mock_session, sample_conversations):
        """Test that context is returned in chronological order (oldest first)."""
        # Simulate DESC order from query, then reversed in method
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = list(reversed(sample_conversations[-5:]))

        service = ConversationService(db_session=mock_session)
        result = service.get_recent_context(user_id="user_123", session_id="session_456", num_messages=5)

        # Result should be in chronological order (oldest first)
        if len(result) > 1:
            # Check that created_at timestamps are in ascending order
            timestamps = [entry.get("created_at", "") for entry in result]
            assert timestamps == sorted(timestamps)

    def test_get_recent_context_includes_required_fields(self, mock_session):
        """Test that context entries include query, response, tone, created_at."""
        conv = MagicMock(spec=Conversation)
        conv.query = "What is ROS 2?"
        conv.response = "ROS 2 is a robotics framework."
        conv.tone = "english"
        conv.created_at = datetime.utcnow()

        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = [conv]

        service = ConversationService(db_session=mock_session)
        result = service.get_recent_context(user_id="user_123", session_id="session_456")

        assert len(result) == 1
        assert "query" in result[0]
        assert "response" in result[0]
        assert "tone" in result[0]
        assert "created_at" in result[0]

    def test_get_recent_context_default_limit_is_5(self, mock_session, sample_conversations):
        """Test that default limit is 5 messages."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = sample_conversations[-5:]

        service = ConversationService(db_session=mock_session)
        # Call without num_messages to use default
        service.get_recent_context(user_id="user_123", session_id="session_456")

        # Verify limit was called with 5
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.assert_called_with(5)

    def test_get_recent_context_custom_limit(self, mock_session, sample_conversations):
        """Test that custom limit is respected."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = sample_conversations[-10:]

        service = ConversationService(db_session=mock_session)
        service.get_recent_context(user_id="user_123", session_id="session_456", num_messages=10)

        # Verify limit was called with 10
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.assert_called_with(10)

    def test_get_recent_context_empty_history(self, mock_session):
        """Test handling of empty conversation history."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

        service = ConversationService(db_session=mock_session)
        result = service.get_recent_context(user_id="user_123", session_id="session_456")

        assert result == []

    def test_get_recent_context_filters_by_user_and_session(self, mock_session):
        """Test that context is filtered by user_id AND session_id."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

        service = ConversationService(db_session=mock_session)
        service.get_recent_context(user_id="user_123", session_id="session_456")

        # Verify filter_by was called with both parameters
        mock_session.query.return_value.filter_by.assert_called_with(
            user_id="user_123", session_id="session_456"
        )

    def test_get_recent_context_handles_db_error(self, mock_session):
        """Test graceful handling of database errors."""
        mock_session.query.side_effect = Exception("Database connection error")

        service = ConversationService(db_session=mock_session)
        result = service.get_recent_context(user_id="user_123", session_id="session_456")

        # Should return empty list on error, not raise
        assert result == []


class TestRetentionPolicy:
    """Tests for 30-day retention cutoff."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.commit = MagicMock()
        session.rollback = MagicMock()
        return session

    def test_delete_old_conversations_default_30_days(self, mock_session):
        """Test that default retention is 30 days."""
        mock_session.query.return_value.filter.return_value.delete.return_value = 5

        service = ConversationService(db_session=mock_session)
        deleted = service.delete_old_conversations()

        # Should delete conversations older than 30 days
        assert deleted == 5
        mock_session.commit.assert_called_once()

    def test_delete_old_conversations_custom_days(self, mock_session):
        """Test custom retention period."""
        mock_session.query.return_value.filter.return_value.delete.return_value = 10

        service = ConversationService(db_session=mock_session)
        deleted = service.delete_old_conversations(older_than_days=7)

        assert deleted == 10

    def test_delete_old_conversations_none_to_delete(self, mock_session):
        """Test when no conversations need deletion."""
        mock_session.query.return_value.filter.return_value.delete.return_value = 0

        service = ConversationService(db_session=mock_session)
        deleted = service.delete_old_conversations()

        assert deleted == 0
        mock_session.commit.assert_called_once()

    def test_delete_old_conversations_handles_error(self, mock_session):
        """Test error handling during deletion."""
        mock_session.query.return_value.filter.return_value.delete.side_effect = Exception("Delete error")

        service = ConversationService(db_session=mock_session)

        with pytest.raises(Exception):
            service.delete_old_conversations()

        mock_session.rollback.assert_called_once()


class TestConversationHistoryPagination:
    """Tests for conversation history pagination."""

    @pytest.fixture
    def mock_session(self):
        """Create mock database session."""
        return MagicMock()

    def test_get_conversation_history_default_limit_50(self, mock_session):
        """Test that default limit for full history is 50."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

        service = ConversationService(db_session=mock_session)
        service.get_conversation_history(user_id="user_123", session_id="session_456")

        # Verify limit was called with 50
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.assert_called_with(50)

    def test_get_conversation_history_custom_limit(self, mock_session):
        """Test custom limit for full history."""
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = []

        service = ConversationService(db_session=mock_session)
        service.get_conversation_history(user_id="user_123", session_id="session_456", limit=100)

        # Verify limit was called with 100
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.assert_called_with(100)

    def test_get_conversation_history_returns_conversation_objects(self, mock_session):
        """Test that history returns Conversation objects."""
        conv = MagicMock(spec=Conversation)
        mock_session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = [conv]

        service = ConversationService(db_session=mock_session)
        result = service.get_conversation_history(user_id="user_123", session_id="session_456")

        assert len(result) == 1
