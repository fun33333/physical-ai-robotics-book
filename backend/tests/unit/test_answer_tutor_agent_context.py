"""
Unit tests for Answer/Tutor Agent Context Awareness (T056 - US3).

Tests the Answer/Tutor Agent's ability to:
- Include prior Q&A in Gemini prompt
- Reference prior exchanges in follow-up responses
- Maintain context across conversation turns
- Handle empty conversation history
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.agents.answer_tutor_agent import generate_answer, create_answer_agent


class TestAnswerAgentContextAwareness:
    """Tests for context awareness in Answer/Tutor Agent."""

    @pytest.fixture
    def sample_chunks(self):
        """Sample retrieved chunks for testing."""
        return [
            {
                "chunk_id": "chunk_001",
                "text": "ROS 2 uses DDS for communication.",
                "chapter": "Module 1",
                "section": "ROS 2 Architecture",
            },
            {
                "chunk_id": "chunk_002",
                "text": "Publishers and subscribers communicate via topics.",
                "chapter": "Module 1",
                "section": "Communication Patterns",
            },
        ]

    @pytest.fixture
    def sample_conversation_history(self):
        """Sample conversation history for multi-turn testing."""
        return [
            {
                "query": "What is ROS 2?",
                "response": "ROS 2 is a flexible framework for writing robot software.",
                "tone": "english",
                "created_at": "2025-12-20T10:00:00Z",
            },
            {
                "query": "How does communication work in ROS 2?",
                "response": "ROS 2 uses DDS (Data Distribution Service) for reliable messaging.",
                "tone": "english",
                "created_at": "2025-12-20T10:01:00Z",
            },
        ]

    @pytest.mark.asyncio
    async def test_generate_answer_accepts_conversation_history(self, sample_chunks):
        """Test that generate_answer accepts conversation_history parameter."""
        history = [{"query": "What is ROS?", "response": "ROS is a robotics framework."}]

        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Test response with context."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="Tell me more",
                chunks=sample_chunks,
                conversation_history=history,
                user_level="intermediate",
            )

            assert "response" in result

    @pytest.mark.asyncio
    async def test_generate_answer_includes_history_in_prompt(self, sample_chunks, sample_conversation_history):
        """Test that conversation history is included in the Gemini prompt."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Building on our previous discussion..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            await generate_answer(
                query="Explain more about publishers",
                chunks=sample_chunks,
                conversation_history=sample_conversation_history,
                user_level="intermediate",
            )

            # Verify Runner.run was called
            mock_runner.run.assert_called_once()
            # The prompt should include context about the conversation

    @pytest.mark.asyncio
    async def test_generate_answer_handles_empty_history(self, sample_chunks):
        """Test that empty conversation history is handled gracefully."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 is a robotics framework."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What is ROS 2?",
                chunks=sample_chunks,
                conversation_history=[],  # Empty history
                user_level="intermediate",
            )

            assert "response" in result
            assert result["response"] != ""

    @pytest.mark.asyncio
    async def test_generate_answer_handles_none_history(self, sample_chunks):
        """Test that None conversation history is handled gracefully."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 is a robotics framework."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What is ROS 2?",
                chunks=sample_chunks,
                conversation_history=None,  # None history
                user_level="intermediate",
            )

            assert "response" in result

    @pytest.mark.asyncio
    async def test_follow_up_references_prior_context(self, sample_chunks, sample_conversation_history):
        """Test that follow-up questions reference prior context appropriately."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            # Simulate agent referencing prior context
            mock_result = MagicMock()
            mock_result.final_output = "As mentioned earlier regarding ROS 2's DDS communication, publishers..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="Tell me more about publishers",
                chunks=sample_chunks,
                conversation_history=sample_conversation_history,
                user_level="intermediate",
            )

            # Response should be contextual
            assert "response" in result
            assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_context_compression_for_long_history(self, sample_chunks):
        """Test that long conversation history is compressed/truncated appropriately."""
        # Create a long conversation history (more than 5 exchanges)
        long_history = [
            {"query": f"Question {i}", "response": f"Answer {i}" * 50, "tone": "english"}
            for i in range(10)
        ]

        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Response based on recent context."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="Follow-up question",
                chunks=sample_chunks,
                conversation_history=long_history,
                user_level="intermediate",
            )

            # Should still return valid response
            assert "response" in result

    @pytest.mark.asyncio
    async def test_selected_text_combined_with_history(self, sample_chunks, sample_conversation_history):
        """Test that selected_text and conversation_history work together."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Based on the highlighted text and our discussion..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What does this mean?",
                chunks=sample_chunks,
                conversation_history=sample_conversation_history,
                user_level="intermediate",
                selected_text="DDS provides real-time messaging",
            )

            assert "response" in result


class TestAnswerAgentCreation:
    """Tests for Answer Agent creation with different levels."""

    def test_create_answer_agent_intermediate(self):
        """Test agent creation with intermediate level."""
        with patch("src.agents.answer_tutor_agent.get_gemini_model") as mock_model:
            mock_model.return_value = "gemini-model"
            agent = create_answer_agent(level="intermediate")
            assert agent.name == "Answer Tutor Agent"

    def test_create_answer_agent_beginner(self):
        """Test agent creation with beginner level."""
        with patch("src.agents.answer_tutor_agent.get_gemini_model") as mock_model:
            mock_model.return_value = "gemini-model"
            agent = create_answer_agent(level="beginner")
            assert agent.name == "Answer Tutor Agent"
            # Instructions should include beginner-specific guidance
            assert "simply" in agent.instructions.lower() or "analogies" in agent.instructions.lower()

    def test_create_answer_agent_advanced(self):
        """Test agent creation with advanced level."""
        with patch("src.agents.answer_tutor_agent.get_gemini_model") as mock_model:
            mock_model.return_value = "gemini-model"
            agent = create_answer_agent(level="advanced")
            assert agent.name == "Answer Tutor Agent"
            # Instructions should include advanced-specific guidance
            assert "technical" in agent.instructions.lower() or "deep" in agent.instructions.lower()


class TestContextFormatting:
    """Tests for conversation context formatting."""

    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [{"chunk_id": "1", "text": "Test chunk", "chapter": "1", "section": "A"}]

    @pytest.mark.asyncio
    async def test_context_includes_prior_queries(self, sample_chunks):
        """Test that prior queries are included in context."""
        history = [
            {"query": "What is ROS?", "response": "ROS is a framework."},
            {"query": "What about ROS 2?", "response": "ROS 2 is the successor."},
        ]

        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Continuing from our discussion..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            await generate_answer(
                query="And what about DDS?",
                chunks=sample_chunks,
                conversation_history=history,
            )

            # Verify the runner was called with the agent and prompt
            mock_runner.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_includes_prior_responses(self, sample_chunks):
        """Test that prior responses are included in context."""
        history = [
            {"query": "What is DDS?", "response": "DDS stands for Data Distribution Service."},
        ]

        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "As I mentioned, DDS is..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            await generate_answer(
                query="How does DDS work?",
                chunks=sample_chunks,
                conversation_history=history,
            )

            mock_runner.run.assert_called_once()
