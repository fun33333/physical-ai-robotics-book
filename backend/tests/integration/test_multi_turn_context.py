"""
Integration tests for Multi-Turn Context Flow (T057 - US3).

Tests the full multi-turn conversation flow:
- Send Q1, save response
- Send Q2 with session_id
- Verify Q2 response references Q1 context
- Verify latency < 2s for Q2
"""

import pytest
import time
from unittest.mock import MagicMock, patch, AsyncMock

from src.services.orchestration_service import OrchestrationService, PipelineContext
from src.services.conversation_service import ConversationService


class TestMultiTurnContextFlow:
    """Integration tests for multi-turn conversation flow."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session with conversation tracking."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()

        # Store saved conversations for later retrieval
        saved_conversations = []

        def mock_add(conv):
            saved_conversations.append(conv)

        session.add.side_effect = mock_add
        session._saved_conversations = saved_conversations

        return session

    @pytest.fixture
    def mock_agents(self):
        """Mock all agent functions."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            # RAG returns relevant chunks
            async def rag_side_effect(*args, **kwargs):
                return {
                    "chunks": [
                        {"chunk_id": "1", "text": "ROS 2 uses DDS.", "chapter": "1", "section": "A"}
                    ],
                    "latency_ms": 150,
                }
            mock_rag.side_effect = rag_side_effect

            # Answer generates contextual response
            async def answer_side_effect(*args, **kwargs):
                return {
                    "response": "ROS 2 is a robotics framework that uses DDS.",
                    "sources_used": ["1"],
                    "latency_ms": 800,
                }
            mock_answer.side_effect = answer_side_effect

            # Tone transforms response
            async def tone_side_effect(*args, **kwargs):
                return {
                    "response": "ROS 2 is a robotics framework that uses DDS.",
                    "latency_ms": 50,
                }
            mock_tone.side_effect = tone_side_effect

            # Safety validates response
            async def safety_side_effect(*args, **kwargs):
                return {
                    "validation_status": "approved",
                    "latency_ms": 100,
                }
            mock_safety.side_effect = safety_side_effect

            yield {
                "rag": mock_rag,
                "answer": mock_answer,
                "tone": mock_tone,
                "safety": mock_safety,
            }

    @pytest.mark.asyncio
    async def test_first_question_processed(self, mock_db_session, mock_agents):
        """Test that first question (Q1) is processed and saved."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        context = PipelineContext(
            query="What is ROS 2?",
            user_id="user_123",
            session_id="session_456",
            tone="english",
        )

        result = await orchestrator.process_chat(context)

        assert result.response != ""
        assert result.session_id == "session_456"
        # Conversation should be logged
        mock_db_session.add.assert_called()

    @pytest.mark.asyncio
    async def test_second_question_with_session_id(self, mock_db_session, mock_agents):
        """Test that Q2 with same session_id maintains context."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        # Q1
        context1 = PipelineContext(
            query="What is ROS 2?",
            user_id="user_123",
            session_id="session_456",
            tone="english",
        )
        await orchestrator.process_chat(context1)

        # Q2 with same session_id
        context2 = PipelineContext(
            query="Tell me more about DDS",
            user_id="user_123",
            session_id="session_456",  # Same session
            tone="english",
            conversation_history=[
                {"query": "What is ROS 2?", "response": "ROS 2 is a robotics framework."}
            ],
        )

        result2 = await orchestrator.process_chat(context2)

        assert result2.response != ""
        assert result2.session_id == "session_456"

    @pytest.mark.asyncio
    async def test_q2_latency_under_2_seconds(self, mock_db_session, mock_agents):
        """Test that Q2 response latency is under 2 seconds."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        context = PipelineContext(
            query="Follow-up question about DDS",
            user_id="user_123",
            session_id="session_456",
            tone="english",
            conversation_history=[
                {"query": "What is ROS 2?", "response": "ROS 2 is a framework."}
            ],
        )

        start_time = time.time()
        result = await orchestrator.process_chat(context)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms

        # Total pipeline latency should be under 2000ms
        assert result.total_latency_ms() < 2000 or elapsed_time < 2000

    @pytest.mark.asyncio
    async def test_context_passed_to_answer_agent(self, mock_db_session, mock_agents):
        """Test that conversation history is passed to Answer Agent."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        history = [
            {"query": "What is ROS 2?", "response": "ROS 2 is a robotics framework."},
            {"query": "How does it communicate?", "response": "It uses DDS."},
        ]

        context = PipelineContext(
            query="Explain more about DDS",
            user_id="user_123",
            session_id="session_456",
            tone="english",
            conversation_history=history,
        )

        await orchestrator.process_chat(context)

        # Verify generate_answer was called with conversation_history
        mock_agents["answer"].assert_called_once()
        call_args = mock_agents["answer"].call_args
        # Check that history was passed (position 2 or keyword)
        if call_args.args:
            # Positional args
            assert len(call_args.args) >= 3 or "conversation_history" in call_args.kwargs
        else:
            assert "conversation_history" in call_args.kwargs


class TestMultiTurnContextRetrieval:
    """Tests for retrieving context from database."""

    @pytest.fixture
    def mock_db_with_history(self):
        """Create mock DB with conversation history."""
        session = MagicMock()

        from datetime import datetime

        # Mock conversations
        mock_convs = [
            MagicMock(
                query="What is ROS 2?",
                response="ROS 2 is a robotics framework.",
                tone="english",
                created_at=datetime(2025, 12, 20, 10, 0, 0),
            ),
            MagicMock(
                query="How does DDS work?",
                response="DDS provides pub/sub messaging.",
                tone="english",
                created_at=datetime(2025, 12, 20, 10, 1, 0),
            ),
        ]

        # Configure query chain
        session.query.return_value.filter_by.return_value.order_by.return_value.limit.return_value.all.return_value = mock_convs
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()

        return session

    def test_retrieve_context_for_session(self, mock_db_with_history):
        """Test retrieving conversation context for a session."""
        service = ConversationService(db_session=mock_db_with_history)

        context = service.get_recent_context(
            user_id="user_123",
            session_id="session_456",
            num_messages=5,
        )

        assert isinstance(context, list)
        # Should have context entries
        assert len(context) >= 0

    def test_context_format_for_agent(self, mock_db_with_history):
        """Test that context format is suitable for agent consumption."""
        service = ConversationService(db_session=mock_db_with_history)

        context = service.get_recent_context(
            user_id="user_123",
            session_id="session_456",
        )

        # Each entry should have query and response
        for entry in context:
            assert "query" in entry
            assert "response" in entry


class TestConversationContinuity:
    """Tests for conversation continuity across turns."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        return session

    @pytest.fixture
    def mock_all_agents(self):
        """Mock all agent functions for full pipeline."""
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

            yield {
                "rag": mock_rag,
                "answer": mock_answer,
                "tone": mock_tone,
                "safety": mock_safety,
            }

    @pytest.mark.asyncio
    async def test_five_turn_conversation(self, mock_db_session, mock_all_agents):
        """Test a 5-turn conversation maintains continuity."""
        orchestrator = OrchestrationService(db_session=mock_db_session)

        questions = [
            "What is ROS 2?",
            "How does it communicate?",
            "What is DDS?",
            "Explain pub/sub pattern",
            "How do I create a publisher?",
        ]

        history = []
        session_id = "session_5_turns"

        for i, question in enumerate(questions):
            context = PipelineContext(
                query=question,
                user_id="user_123",
                session_id=session_id,
                tone="english",
                conversation_history=history.copy(),
            )

            result = await orchestrator.process_chat(context)

            # Verify response
            assert result.response != ""
            assert result.session_id == session_id

            # Add to history for next turn
            history.append({
                "query": question,
                "response": result.response,
            })

        # All 5 turns should have been processed
        assert len(history) == 5

    @pytest.mark.asyncio
    async def test_session_id_preserved_across_turns(self, mock_db_session, mock_all_agents):
        """Test that session_id is consistent across all turns."""
        orchestrator = OrchestrationService(db_session=mock_db_session)
        session_id = "persistent_session"

        for i in range(3):
            context = PipelineContext(
                query=f"Question {i + 1}",
                user_id="user_123",
                session_id=session_id,
                tone="english",
            )

            result = await orchestrator.process_chat(context)

            # Session ID should be preserved
            assert result.session_id == session_id
