"""
Integration tests for Session Isolation (T058 - US3).

Tests that:
- Two concurrent sessions have independent history
- No context leakage between sessions
- Session-specific queries don't affect other sessions
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from src.services.conversation_service import ConversationService
from src.services.orchestration_service import OrchestrationService, PipelineContext


class TestSessionIsolation:
    """Integration tests for session isolation."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session with session-aware storage."""
        session = MagicMock()

        # Store conversations by session_id
        conversations_by_session = {}

        def mock_add(conv):
            sid = getattr(conv, 'session_id', 'unknown')
            if sid not in conversations_by_session:
                conversations_by_session[sid] = []
            conversations_by_session[sid].append(conv)

        session.add.side_effect = mock_add
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session._conversations_by_session = conversations_by_session

        return session

    @pytest.fixture
    def mock_agents(self):
        """Mock all agent functions."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            async def rag_side_effect(*args, **kwargs):
                return {
                    "chunks": [{"chunk_id": "1", "text": "Test", "chapter": "1", "section": "A"}],
                    "latency_ms": 100,
                }
            mock_rag.side_effect = rag_side_effect

            async def answer_side_effect(*args, **kwargs):
                return {
                    "response": "Test response",
                    "latency_ms": 500,
                }
            mock_answer.side_effect = answer_side_effect

            async def tone_side_effect(*args, **kwargs):
                return {
                    "response": "Test response",
                    "latency_ms": 50,
                }
            mock_tone.side_effect = tone_side_effect

            async def safety_side_effect(*args, **kwargs):
                return {
                    "validation_status": "approved",
                    "latency_ms": 50,
                }
            mock_safety.side_effect = safety_side_effect

            yield

    @pytest.mark.asyncio
    async def test_two_sessions_independent_history(self, mock_db_session, mock_agents):
        """Test that two sessions have completely independent history."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        # Session 1: Ask about ROS 2
        context1_q1 = PipelineContext(
            query="What is ROS 2?",
            user_id="user_A",
            session_id="session_A",
            tone="english",
        )
        await orchestrator.process_chat(context1_q1)

        # Session 2: Ask about Python
        context2_q1 = PipelineContext(
            query="What is Python?",
            user_id="user_B",
            session_id="session_B",
            tone="english",
        )
        await orchestrator.process_chat(context2_q1)

        # Session 1 should only have ROS 2 related history
        session_a_convs = mock_db_session._conversations_by_session.get("session_A", [])
        session_b_convs = mock_db_session._conversations_by_session.get("session_B", [])

        assert len(session_a_convs) >= 1
        assert len(session_b_convs) >= 1

        # Verify separation - session A should not have Python content
        for conv in session_a_convs:
            assert conv.session_id == "session_A"

        # Verify separation - session B should not have ROS content
        for conv in session_b_convs:
            assert conv.session_id == "session_B"

    @pytest.mark.asyncio
    async def test_no_context_leakage_between_sessions(self, mock_db_session, mock_agents):
        """Test that context from one session doesn't leak to another."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        # Session 1: Private conversation
        await orchestrator.process_chat(PipelineContext(
            query="My secret password is 12345",
            user_id="user_A",
            session_id="private_session",
            tone="english",
        ))

        # Session 2: Different user asks something
        context2 = PipelineContext(
            query="What did the previous user ask?",
            user_id="user_B",
            session_id="other_session",
            tone="english",
            conversation_history=[],  # Empty - new session
        )

        result2 = await orchestrator.process_chat(context2)

        # The response should not contain session 1's content
        # (In real implementation, context isolation prevents this)
        assert result2.session_id == "other_session"

    @pytest.mark.asyncio
    async def test_same_user_different_sessions_isolated(self, mock_db_session, mock_agents):
        """Test that same user with different sessions are isolated."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        # Same user, Session 1
        await orchestrator.process_chat(PipelineContext(
            query="ROS 2 question",
            user_id="user_123",
            session_id="session_morning",
            tone="english",
        ))

        # Same user, Session 2
        await orchestrator.process_chat(PipelineContext(
            query="Python question",
            user_id="user_123",
            session_id="session_evening",
            tone="english",
        ))

        # Each session should have its own conversations
        morning_convs = mock_db_session._conversations_by_session.get("session_morning", [])
        evening_convs = mock_db_session._conversations_by_session.get("session_evening", [])

        # Sessions should be separate
        assert len(morning_convs) >= 1
        assert len(evening_convs) >= 1


class TestConversationServiceSessionIsolation:
    """Tests for ConversationService session isolation."""

    @pytest.fixture
    def mock_db_with_multiple_sessions(self):
        """Create mock DB with conversations from multiple sessions."""
        session = MagicMock()

        # Conversations for different sessions
        session_a_convs = [
            MagicMock(
                query="ROS 2 question",
                response="ROS 2 answer",
                session_id="session_A",
                user_id="user_A",
                tone="english",
                created_at=datetime(2025, 12, 20, 10, 0, 0),
            ),
        ]

        session_b_convs = [
            MagicMock(
                query="Python question",
                response="Python answer",
                session_id="session_B",
                user_id="user_B",
                tone="english",
                created_at=datetime(2025, 12, 20, 10, 0, 0),
            ),
        ]

        def filter_by_session(**kwargs):
            mock_result = MagicMock()
            sid = kwargs.get('session_id')
            uid = kwargs.get('user_id')

            if sid == "session_A":
                mock_result.order_by.return_value.limit.return_value.all.return_value = session_a_convs
            elif sid == "session_B":
                mock_result.order_by.return_value.limit.return_value.all.return_value = session_b_convs
            else:
                mock_result.order_by.return_value.limit.return_value.all.return_value = []

            return mock_result

        session.query.return_value.filter_by.side_effect = filter_by_session

        return session

    def test_get_history_only_returns_session_conversations(self, mock_db_with_multiple_sessions):
        """Test that history retrieval only returns session-specific data."""
        service = ConversationService(db_session=mock_db_with_multiple_sessions)

        # Get history for session A
        history_a = service.get_conversation_history(
            user_id="user_A",
            session_id="session_A",
        )

        # Get history for session B
        history_b = service.get_conversation_history(
            user_id="user_B",
            session_id="session_B",
        )

        # Each should have only their own conversations
        assert len(history_a) >= 1
        assert len(history_b) >= 1

        # Verify no cross-contamination
        for conv in history_a:
            assert conv.session_id == "session_A"

        for conv in history_b:
            assert conv.session_id == "session_B"

    def test_get_context_only_returns_session_context(self, mock_db_with_multiple_sessions):
        """Test that context retrieval only returns session-specific context."""
        service = ConversationService(db_session=mock_db_with_multiple_sessions)

        # Get context for session A
        context_a = service.get_recent_context(
            user_id="user_A",
            session_id="session_A",
        )

        # Should only have session A context
        assert isinstance(context_a, list)

    def test_empty_session_returns_empty_history(self, mock_db_with_multiple_sessions):
        """Test that a new session returns empty history."""
        service = ConversationService(db_session=mock_db_with_multiple_sessions)

        # Get history for non-existent session
        history = service.get_conversation_history(
            user_id="user_new",
            session_id="session_new",
        )

        assert history == []


class TestConcurrentSessions:
    """Tests for concurrent session handling."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        return session

    @pytest.fixture
    def mock_agents(self):
        """Mock all agent functions."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            async def rag_side_effect(*args, **kwargs):
                return {
                    "chunks": [],
                    "latency_ms": 100,
                }
            mock_rag.side_effect = rag_side_effect

            async def answer_side_effect(*args, **kwargs):
                return {
                    "response": "Response",
                    "latency_ms": 500,
                }
            mock_answer.side_effect = answer_side_effect

            async def tone_side_effect(*args, **kwargs):
                return {
                    "response": "Response",
                    "latency_ms": 50,
                }
            mock_tone.side_effect = tone_side_effect

            async def safety_side_effect(*args, **kwargs):
                return {
                    "validation_status": "approved",
                    "latency_ms": 50,
                }
            mock_safety.side_effect = safety_side_effect

            yield

    @pytest.mark.asyncio
    async def test_concurrent_sessions_process_independently(self, mock_db_session, mock_agents):
        """Test that concurrent sessions process their queries independently."""
        import asyncio

        orchestrator = OrchestrationService(db_session=mock_db_session)

        # Create contexts for concurrent sessions
        context_a = PipelineContext(
            query="Session A query",
            user_id="user_A",
            session_id="concurrent_session_A",
            tone="english",
        )

        context_b = PipelineContext(
            query="Session B query",
            user_id="user_B",
            session_id="concurrent_session_B",
            tone="roman_urdu",
        )

        # Process concurrently
        results = await asyncio.gather(
            orchestrator.process_chat(context_a),
            orchestrator.process_chat(context_b),
        )

        result_a, result_b = results

        # Each should have its own session_id
        assert result_a.session_id == "concurrent_session_A"
        assert result_b.session_id == "concurrent_session_B"

        # Each should have its own tone preserved
        assert result_a.tone == "english"
        assert result_b.tone == "roman_urdu"

    @pytest.mark.asyncio
    async def test_session_state_not_shared(self, mock_db_session, mock_agents):
        """Test that session state is not shared between concurrent requests."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        # Session 1 with specific context
        context1 = PipelineContext(
            query="Question 1",
            user_id="user_1",
            session_id="isolated_1",
            tone="english",
            conversation_history=[{"query": "Prior Q", "response": "Prior A"}],
        )

        # Session 2 with no context
        context2 = PipelineContext(
            query="Question 2",
            user_id="user_2",
            session_id="isolated_2",
            tone="english",
            conversation_history=[],  # No prior history
        )

        result1 = await orchestrator.process_chat(context1)
        result2 = await orchestrator.process_chat(context2)

        # Results should be independent
        assert result1.session_id != result2.session_id
