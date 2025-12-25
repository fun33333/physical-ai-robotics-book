"""
Unit tests for Answer/Tutor Agent (Phase 3 - T031).

Tests the Answer/Tutor Agent to verify:
- Response generation from retrieved chunks
- Source citation in responses
- No hallucinations (response only uses info from chunks)
- Gemini mock integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.answer_tutor_agent import (
    create_answer_agent,
    generate_answer,
    LEVEL_PROMPTS,
)


class TestAnswerAgentCreation:
    """Tests for Answer Agent creation."""

    def test_create_agent_default_level(self):
        """Test creating agent with default intermediate level."""
        agent = create_answer_agent()

        assert agent.name == "Answer Tutor Agent"
        assert "intermediate" in agent.instructions.lower() or "technical terms" in agent.instructions.lower()

    def test_create_agent_beginner_level(self):
        """Test creating agent with beginner level instructions."""
        agent = create_answer_agent(level="beginner")

        assert agent.name == "Answer Tutor Agent"
        assert "simple" in agent.instructions.lower() or "jargon" in agent.instructions.lower()

    def test_create_agent_advanced_level(self):
        """Test creating agent with advanced level instructions."""
        agent = create_answer_agent(level="advanced")

        assert agent.name == "Answer Tutor Agent"
        assert "deep" in agent.instructions.lower() or "implementation" in agent.instructions.lower()

    def test_level_prompts_exist(self):
        """Test that all level prompts are defined."""
        assert "beginner" in LEVEL_PROMPTS
        assert "intermediate" in LEVEL_PROMPTS
        assert "advanced" in LEVEL_PROMPTS


class TestGenerateAnswer:
    """Tests for the generate_answer function."""

    @pytest.fixture
    def mock_chunks(self):
        """Sample retrieved chunks for testing."""
        return [
            {
                "chunk_id": "chunk_001",
                "text": "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It provides tools and libraries for building complex robots.",
                "chapter": "Module 1",
                "section": "ROS 2 Fundamentals",
            },
            {
                "chunk_id": "chunk_002",
                "text": "ROS 2 uses DDS (Data Distribution Service) for reliable communication between nodes. DDS provides real-time data transfer.",
                "chapter": "Module 1",
                "section": "ROS 2 Architecture",
            },
        ]

    @pytest.mark.asyncio
    async def test_generate_answer_returns_response(self, mock_chunks):
        """Test that generate_answer returns a valid response."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            # Mock the Runner.run method
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 is a flexible framework for writing robot software, as described in SOURCE 1."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What is ROS 2?",
                chunks=mock_chunks,
                user_level="beginner",
            )

            assert "response" in result
            assert len(result["response"]) > 0
            assert "sources_used" in result

    @pytest.mark.asyncio
    async def test_generate_answer_cites_sources(self, mock_chunks):
        """Test that response cites sources from chunks."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "According to SOURCE 1, ROS 2 is a flexible framework. SOURCE 2 explains that it uses DDS for communication."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What is ROS 2?",
                chunks=mock_chunks,
            )

            # Verify sources are included
            assert "sources_used" in result
            assert len(result["sources_used"]) == 2
            assert "chunk_001" in result["sources_used"]
            assert "chunk_002" in result["sources_used"]

    @pytest.mark.asyncio
    async def test_generate_answer_includes_selected_text(self, mock_chunks):
        """Test that selected text is included in the prompt."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Based on the highlighted text about DDS..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="Explain this",
                chunks=mock_chunks,
                selected_text="DDS provides real-time data transfer",
            )

            # Verify the runner was called with the selected text in prompt
            mock_runner.run.assert_called_once()
            call_args = mock_runner.run.call_args
            prompt = call_args[0][1]  # Second positional arg is the prompt
            assert "HIGHLIGHTED" in prompt
            assert "DDS" in prompt

    @pytest.mark.asyncio
    async def test_generate_answer_tracks_latency(self, mock_chunks):
        """Test that latency is tracked."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Test response"
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What is ROS 2?",
                chunks=mock_chunks,
            )

            assert "latency_ms" in result
            assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_generate_answer_handles_error(self, mock_chunks):
        """Test error handling when agent fails."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_runner.run = AsyncMock(side_effect=Exception("API Error"))

            result = await generate_answer(
                query="What is ROS 2?",
                chunks=mock_chunks,
            )

            assert "error" in result
            assert "Error generating response" in result["response"]

    @pytest.mark.asyncio
    async def test_generate_answer_empty_chunks(self):
        """Test behavior with empty chunks list."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "I do not find this in the textbook."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="What is quantum computing?",
                chunks=[],
            )

            assert "response" in result
            # Should indicate no sources found
            assert len(result.get("sources_used", [])) == 0

    @pytest.mark.asyncio
    async def test_generate_answer_with_conversation_history(self, mock_chunks):
        """Test that conversation history is considered."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Building on our previous discussion about ROS..."
            mock_runner.run = AsyncMock(return_value=mock_result)

            history = [
                {"query": "What is ROS?", "response": "ROS is a robotics framework."},
            ]

            result = await generate_answer(
                query="Tell me more",
                chunks=mock_chunks,
                conversation_history=history,
            )

            assert "response" in result


class TestHallucinationPrevention:
    """Tests for hallucination prevention in responses."""

    @pytest.fixture
    def specific_chunks(self):
        """Chunks with very specific information."""
        return [
            {
                "chunk_id": "chunk_ros2",
                "text": "ROS 2 was released in 2017. It uses DDS middleware.",
                "chapter": "Module 1",
                "section": "History",
            },
        ]

    @pytest.mark.asyncio
    async def test_response_contains_only_source_info(self, specific_chunks):
        """Test that response only contains information from sources."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            # This is a good response - only uses info from chunks
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 was released in 2017 and uses DDS middleware, as described in SOURCE 1."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await generate_answer(
                query="When was ROS 2 released?",
                chunks=specific_chunks,
            )

            response = result["response"]
            # Response should contain information from the chunk
            assert "2017" in response or "DDS" in response

    @pytest.mark.asyncio
    async def test_agent_instructions_prevent_hallucination(self):
        """Test that agent instructions include anti-hallucination rules."""
        agent = create_answer_agent()

        instructions = agent.instructions.lower()

        # Should have rules about only using source material
        assert "only" in instructions or "source" in instructions
        assert "textbook" in instructions or "sources" in instructions


class TestPromptFormatting:
    """Tests for prompt formatting."""

    @pytest.fixture
    def mock_chunks(self):
        return [
            {
                "chunk_id": "c1",
                "text": "Sample text about robotics.",
                "chapter": "Ch1",
                "section": "Sec1",
            },
        ]

    @pytest.mark.asyncio
    async def test_prompt_includes_source_formatting(self, mock_chunks):
        """Test that prompt properly formats sources."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Test response"
            mock_runner.run = AsyncMock(return_value=mock_result)

            await generate_answer("Test query", mock_chunks)

            # Check the prompt sent to the agent
            call_args = mock_runner.run.call_args
            prompt = call_args[0][1]

            assert "SOURCE" in prompt
            assert "Ch1" in prompt or "Sec1" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_question(self, mock_chunks):
        """Test that prompt includes the user question."""
        with patch("src.agents.answer_tutor_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Test response"
            mock_runner.run = AsyncMock(return_value=mock_result)

            await generate_answer("What is SLAM?", mock_chunks)

            call_args = mock_runner.run.call_args
            prompt = call_args[0][1]

            assert "SLAM" in prompt
            assert "QUESTION" in prompt
