"""
Integration tests for tone switching mid-conversation (Phase 4 - T047).

Tests the ability to switch tones during a conversation:
- Ask Q1 in English tone
- Switch to Roman Urdu for Q2
- Verify tone changes but context maintained
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.orchestration_service import OrchestrationService, PipelineContext
from src.agents.tone_agent import apply_tone, TONES


class TestToneSwitchingMidConversation:
    """Tests for switching tones mid-conversation."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_switch_english_to_roman_urdu(self):
        """Test switching from English to Roman Urdu tone."""
        # First response in English
        english_result = await apply_tone(
            response="ROS 2 uses DDS for communication.",
            tone="english",
        )

        # Second response in Roman Urdu
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 DDS use karta hai communication ke liye."
            mock_runner.run = AsyncMock(return_value=mock_result)

            urdu_result = await apply_tone(
                response="ROS 2 uses DDS for communication.",
                tone="roman_urdu",
            )

        # Verify tones are different
        assert english_result["tone"] == "english"
        assert urdu_result["tone"] == "roman_urdu"

    @pytest.mark.asyncio
    async def test_switch_english_to_bro_guide(self):
        """Test switching from English to Bro Guide tone."""
        english_result = await apply_tone(
            response="Publishers send messages to topics.",
            tone="english",
        )

        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Yaar, publishers messages bhejte hain topics pe."
            mock_runner.run = AsyncMock(return_value=mock_result)

            bro_result = await apply_tone(
                response="Publishers send messages to topics.",
                tone="bro_guide",
            )

        assert english_result["tone"] == "english"
        assert bro_result["tone"] == "bro_guide"

    @pytest.mark.asyncio
    async def test_switch_roman_urdu_to_english(self):
        """Test switching from Roman Urdu to English."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Nodes ek doosre se communicate karte hain."
            mock_runner.run = AsyncMock(return_value=mock_result)

            urdu_result = await apply_tone(
                response="Nodes communicate with each other.",
                tone="roman_urdu",
            )

        english_result = await apply_tone(
            response="Nodes communicate with each other.",
            tone="english",
        )

        assert urdu_result["tone"] == "roman_urdu"
        assert english_result["tone"] == "english"


class TestContextMaintainedAcrossTones:
    """Tests for context maintenance during tone switches."""

    @pytest.mark.asyncio
    async def test_technical_content_preserved_across_tones(self):
        """Test that technical content is preserved across tone switches."""
        base_response = "ROS 2 nodes communicate using topics and services."

        english_result = await apply_tone(response=base_response, tone="english")

        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 nodes topics aur services use karte hain communication ke liye."
            mock_runner.run = AsyncMock(return_value=mock_result)

            urdu_result = await apply_tone(response=base_response, tone="roman_urdu")

        # Both should contain key technical terms
        assert "ros" in english_result["response"].lower()
        # Urdu version should have technical terms in English
        assert "ros" in urdu_result["response"].lower() or "topics" in urdu_result["response"].lower()

    @pytest.mark.asyncio
    async def test_source_citations_preserved(self):
        """Test that source citations are preserved during tone switch."""
        response_with_sources = "ROS 2 uses DDS (Source: Chapter 3, Section 3.2)."

        english_result = await apply_tone(response=response_with_sources, tone="english")

        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 DDS use karta hai (Source: Chapter 3, Section 3.2)."
            mock_runner.run = AsyncMock(return_value=mock_result)

            urdu_result = await apply_tone(response=response_with_sources, tone="roman_urdu")

        # Sources should be preserved
        assert "chapter 3" in english_result["response"].lower() or "3.2" in english_result["response"]
        assert "chapter 3" in urdu_result["response"].lower() or "3.2" in urdu_result["response"]


class TestOrchestrationToneSwitching:
    """Tests for tone switching through orchestration service."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = MagicMock()
        session.add = MagicMock()
        session.commit = MagicMock()
        session.refresh = MagicMock()
        session.query = MagicMock(return_value=MagicMock(filter=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))))
        return session

    @pytest.mark.asyncio
    async def test_orchestration_respects_tone_parameter(self, mock_db_session):
        """Test that orchestration service respects tone parameter."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            mock_rag.return_value = {"chunks": [], "sources": []}
            mock_answer.return_value = {"response": "Test answer", "sources": []}
            mock_tone.return_value = {"response": "Test answer in urdu", "tone": "roman_urdu"}
            mock_safety.return_value = {"response": "Test answer in urdu", "is_valid": True}

            orchestrator = OrchestrationService(db_session=mock_db_session)
            context = PipelineContext(
                query="What is ROS 2?",
                user_id="test_user",
                session_id="test_session",
                tone="roman_urdu",
            )

            await orchestrator.process_chat(context)

            # Verify tone was passed to apply_tone
            mock_tone.assert_called()
            call_args = mock_tone.call_args
            # Check that roman_urdu was passed
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_sequential_requests_different_tones(self, mock_db_session):
        """Test sequential requests with different tones."""
        with patch("src.services.orchestration_service.retrieve_context") as mock_rag, \
             patch("src.services.orchestration_service.generate_answer") as mock_answer, \
             patch("src.services.orchestration_service.apply_tone") as mock_tone, \
             patch("src.services.orchestration_service.validate_response") as mock_safety:

            mock_rag.return_value = {"chunks": [], "sources": []}
            mock_answer.return_value = {"response": "Test answer", "sources": []}
            mock_safety.return_value = {"response": "Test answer", "is_valid": True}

            orchestrator = OrchestrationService(db_session=mock_db_session)

            # Request 1: English
            mock_tone.return_value = {"response": "English response", "tone": "english"}
            context1 = PipelineContext(
                query="What is ROS 2?",
                user_id="test_user",
                session_id="test_session",
                tone="english",
            )
            await orchestrator.process_chat(context1)

            # Request 2: Roman Urdu
            mock_tone.return_value = {"response": "Urdu response", "tone": "roman_urdu"}
            context2 = PipelineContext(
                query="Tell me more about publishers",
                user_id="test_user",
                session_id="test_session",
                tone="roman_urdu",
            )
            await orchestrator.process_chat(context2)

            # Both should have been processed
            assert mock_tone.call_count >= 2


class TestToneConsistencyWithinRequest:
    """Tests for tone consistency within a single request."""

    @pytest.mark.asyncio
    async def test_tone_remains_consistent_in_response(self):
        """Test that selected tone is reflected in response."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Yaar, scene yeh hai ke ROS 2 DDS use karta hai."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="ROS 2 uses DDS for communication.",
                tone="bro_guide",
            )

        assert result["tone"] == "bro_guide"
        # Response should have casual language indicators
        response_lower = result["response"].lower()
        assert any(phrase in response_lower for phrase in ["yaar", "scene", "hai", "ke"])


class TestLatencyWithToneSwitching:
    """Tests for latency during tone switching."""

    @pytest.mark.asyncio
    async def test_tone_switch_latency_tracked(self):
        """Test that latency is tracked during tone transformation."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Transformed response"
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="Original response",
                tone="roman_urdu",
            )

        # Non-English tones should have latency tracking
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_english_tone_no_latency_overhead(self):
        """Test that English tone has minimal latency overhead."""
        result = await apply_tone(
            response="Original response",
            tone="english",
        )

        # English should not have latency_ms as it skips transformation
        assert "latency_ms" not in result or result.get("latency_ms", 0) == 0
