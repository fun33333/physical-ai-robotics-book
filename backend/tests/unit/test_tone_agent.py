"""
Unit tests for Tone Agent (Phase 3 - T034).

Tests the Tone Agent to verify:
- Tone transformation (english, roman_urdu, bro_guide)
- Original response preservation
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.tone_agent import (
    create_tone_agent,
    apply_tone,
    TONES,
)


class TestToneAgentConfiguration:
    """Tests for Tone Agent configuration."""

    def test_create_tone_agent_english(self):
        """Test creating English tone agent."""
        agent = create_tone_agent("english")
        assert agent is not None
        assert "english" in agent.name.lower() or "Tone Agent" in agent.name

    def test_create_tone_agent_roman_urdu(self):
        """Test creating Roman Urdu tone agent."""
        agent = create_tone_agent("roman_urdu")
        assert agent is not None

    def test_create_tone_agent_bro_guide(self):
        """Test creating Bro Guide tone agent."""
        agent = create_tone_agent("bro_guide")
        assert agent is not None

    def test_agent_has_instructions(self):
        """Test that agent has tone instructions."""
        agent = create_tone_agent("english")
        assert agent.instructions is not None
        assert len(agent.instructions) > 0

    def test_tones_constant_exists(self):
        """Test that TONES is defined."""
        assert TONES is not None
        assert isinstance(TONES, dict)

    def test_all_tones_defined(self):
        """Test that all supported tones have instructions."""
        required_tones = ["english", "roman_urdu", "bro_guide"]

        for tone in required_tones:
            assert tone in TONES


class TestApplyTone:
    """Tests for the apply_tone function."""

    @pytest.mark.asyncio
    async def test_apply_english_tone(self):
        """Test applying English (formal) tone."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 is a robotics framework that was released in December 2017."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="ROS 2 is a robotics framework released in 2017.",
                tone="english",
            )

            assert "response" in result
            assert len(result["response"]) > 0

    @pytest.mark.asyncio
    async def test_apply_roman_urdu_tone(self):
        """Test applying Roman Urdu tone."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 ek robotics framework hai jo 2017 mein release hua."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="ROS 2 is a robotics framework released in 2017.",
                tone="roman_urdu",
            )

            assert "response" in result

    @pytest.mark.asyncio
    async def test_apply_bro_guide_tone(self):
        """Test applying Bro Guide (casual) tone."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Yo bro, ROS 2 is like this dope robotics framework from 2017."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="ROS 2 is a robotics framework released in 2017.",
                tone="bro_guide",
            )

            assert "response" in result

    @pytest.mark.asyncio
    async def test_tracks_latency_for_non_english(self):
        """Test that latency is tracked for non-English tones."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Test response in Urdu"
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="Test",
                tone="roman_urdu",  # Non-English tone triggers LLM call
            )

            assert "latency_ms" in result
            assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_english_tone_skips_llm(self):
        """Test that English tone skips LLM transformation."""
        result = await apply_tone(
            response="Test response",
            tone="english",
        )

        # English tone should return response directly
        assert result["response"] == "Test response"
        assert result["tone"] == "english"

    @pytest.mark.asyncio
    async def test_preserves_original_on_error(self):
        """Test that original response is preserved on error."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_runner.run = AsyncMock(side_effect=Exception("API Error"))

            original_response = "Original test response about ROS 2."

            result = await apply_tone(
                response=original_response,
                tone="english",
            )

            # Should return the original response on error
            assert result["response"] == original_response

    @pytest.mark.asyncio
    async def test_unknown_tone_defaults_to_english(self):
        """Test that unknown tones default to English."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = "Test response in English."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="Test response",
                tone="unknown_tone",
            )

            assert "response" in result


class TestTonePreservation:
    """Tests for content preservation across tone transformation."""

    @pytest.mark.asyncio
    async def test_preserves_factual_content(self):
        """Test that factual content is preserved in toned response."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            # The toned response should still contain the key facts
            mock_result = MagicMock()
            mock_result.final_output = "ROS 2 was officially released in December 2017 with DDS support."
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="ROS 2 was released in December 2017 using DDS.",
                tone="english",
            )

            # Check that key terms are preserved
            response_lower = result["response"].lower()
            assert "ros 2" in response_lower or "ros" in response_lower
            assert "2017" in response_lower
            assert "dds" in response_lower

    @pytest.mark.asyncio
    async def test_handles_empty_response(self):
        """Test handling of empty response."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            mock_result = MagicMock()
            mock_result.final_output = ""
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response="",
                tone="english",
            )

            assert "response" in result

    @pytest.mark.asyncio
    async def test_handles_long_response(self):
        """Test handling of long responses."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            long_response = "ROS 2 provides many features. " * 100
            mock_result = MagicMock()
            mock_result.final_output = long_response
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response=long_response,
                tone="english",
            )

            assert "response" in result
            assert len(result["response"]) > 0


class TestToneInstructionsContent:
    """Tests for the content of tone instructions."""

    def test_english_tone_is_formal(self):
        """Test that English tone instructions emphasize formality."""
        english_instructions = TONES.get("english", "").lower()

        assert "formal" in english_instructions or "academic" in english_instructions or "clear" in english_instructions

    def test_roman_urdu_tone_mentions_urdu(self):
        """Test that Roman Urdu instructions mention Urdu."""
        roman_urdu_instructions = TONES.get("roman_urdu", "").lower()

        assert "urdu" in roman_urdu_instructions or "roman" in roman_urdu_instructions

    def test_bro_guide_tone_is_casual(self):
        """Test that Bro Guide instructions emphasize casual style."""
        bro_guide_instructions = TONES.get("bro_guide", "").lower()

        assert "casual" in bro_guide_instructions or "karachi" in bro_guide_instructions or "bro" in bro_guide_instructions
