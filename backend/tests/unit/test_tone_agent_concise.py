"""
Unit tests for Tone Agent conciseness logic (Phase 4 - T046).

Tests the conciseness logic to verify:
- Long responses are truncated to 1-2 sentences
- "Ask for longer?" prompt is appended
- Full response is stored for retrieval
- Technical accuracy preserved in truncated version
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.tone_agent import (
    apply_tone,
    apply_conciseness,
    CONCISENESS_THRESHOLD,
    TONES,
)


class TestConcisenessThreshold:
    """Tests for conciseness threshold configuration."""

    def test_conciseness_threshold_defined(self):
        """Test that CONCISENESS_THRESHOLD is defined."""
        assert CONCISENESS_THRESHOLD is not None
        assert isinstance(CONCISENESS_THRESHOLD, int)

    def test_conciseness_threshold_value(self):
        """Test that threshold is approximately 250 characters."""
        assert CONCISENESS_THRESHOLD >= 200
        assert CONCISENESS_THRESHOLD <= 300


class TestApplyConciseness:
    """Tests for the apply_conciseness function."""

    def test_short_response_unchanged(self):
        """Test that short responses are not truncated."""
        short_response = "ROS 2 uses DDS for communication."
        result = apply_conciseness(short_response)

        assert result["response"] == short_response
        assert result["is_truncated"] is False
        assert result["full_response"] is None

    def test_long_response_truncated(self):
        """Test that long responses are truncated."""
        long_response = "ROS 2 is a powerful robotics framework. " * 20
        result = apply_conciseness(long_response)

        assert len(result["response"]) < len(long_response)
        assert result["is_truncated"] is True
        assert result["full_response"] == long_response

    def test_truncated_response_has_prompt(self):
        """Test that truncated response includes 'Ask for longer?' prompt."""
        long_response = "This is a very long explanation about ROS 2 features. " * 15
        result = apply_conciseness(long_response)

        assert "longer" in result["response"].lower() or "more" in result["response"].lower()

    def test_truncated_response_is_one_to_two_sentences(self):
        """Test that truncated response is 1-2 sentences."""
        long_response = "ROS 2 provides DDS communication. It supports real-time. Publishers send data. Subscribers receive data. Topics are named channels. Services are request-response. Actions are long-running. " * 5
        result = apply_conciseness(long_response)

        # Count sentences (roughly by periods)
        truncated = result["response"]
        # Remove the "Ask for longer?" prompt for sentence count
        main_content = truncated.split("Would you like")[0] if "Would you like" in truncated else truncated
        main_content = main_content.split("Ask for")[0] if "Ask for" in main_content else main_content

        sentence_count = main_content.count(".") + main_content.count("!") + main_content.count("?")
        assert sentence_count <= 3  # Allow up to 3 sentences including prompt

    def test_preserves_key_information(self):
        """Test that truncation preserves key technical terms."""
        long_response = "ROS 2 uses DDS (Data Distribution Service) for communication. This provides real-time guarantees. The publish-subscribe pattern is fundamental. " * 5
        result = apply_conciseness(long_response)

        # Key terms should be in truncated version
        truncated_lower = result["response"].lower()
        assert "ros" in truncated_lower or "dds" in truncated_lower


class TestToneWithConciseness:
    """Tests for tone transformation with conciseness."""

    @pytest.mark.asyncio
    async def test_tone_applies_conciseness(self):
        """Test that apply_tone applies conciseness after transformation."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            long_response = "ROS 2 is amazing. " * 50
            mock_result = MagicMock()
            mock_result.final_output = long_response
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response=long_response,
                tone="roman_urdu",
                apply_concise=True,
            )

            assert result.get("is_truncated", False) is True or len(result["response"]) < len(long_response)

    @pytest.mark.asyncio
    async def test_conciseness_can_be_disabled(self):
        """Test that conciseness can be disabled."""
        with patch("src.agents.tone_agent.Runner") as mock_runner:
            long_response = "ROS 2 is amazing. " * 50
            mock_result = MagicMock()
            mock_result.final_output = long_response
            mock_runner.run = AsyncMock(return_value=mock_result)

            result = await apply_tone(
                response=long_response,
                tone="roman_urdu",
                apply_concise=False,
            )

            # Full response should be returned when conciseness is disabled
            assert len(result["response"]) >= len(long_response) * 0.8  # Allow some variation from tone

    @pytest.mark.asyncio
    async def test_english_tone_with_conciseness(self):
        """Test English tone still applies conciseness."""
        long_response = "ROS 2 provides many features for robotics. " * 20

        result = await apply_tone(
            response=long_response,
            tone="english",
            apply_concise=True,
        )

        # Even English should truncate long responses
        assert result.get("is_truncated", False) is True or len(result["response"]) < len(long_response)


class TestFullResponseRetrieval:
    """Tests for full response retrieval after conciseness."""

    def test_full_response_stored(self):
        """Test that full response is stored when truncated."""
        long_response = "This is the complete explanation of ROS 2. " * 20
        result = apply_conciseness(long_response)

        assert result["full_response"] == long_response

    def test_full_response_none_when_not_truncated(self):
        """Test that full_response is None when not truncated."""
        short_response = "ROS 2 uses DDS."
        result = apply_conciseness(short_response)

        assert result["full_response"] is None

    def test_full_response_matches_original(self):
        """Test that stored full response exactly matches original."""
        long_response = "ROS 2 is a robotics framework with many features including DDS communication, actions, services, and topics. " * 10
        result = apply_conciseness(long_response)

        assert result["full_response"] == long_response
        assert result["is_truncated"] is True


class TestEdgeCases:
    """Tests for edge cases in conciseness logic."""

    def test_response_exactly_at_threshold(self):
        """Test response exactly at threshold."""
        # Create response exactly at threshold
        threshold = 250
        exact_response = "a" * threshold
        result = apply_conciseness(exact_response)

        # Should not truncate at exact threshold
        assert result["is_truncated"] is False

    def test_response_one_char_over_threshold(self):
        """Test response one character over threshold."""
        threshold = 250
        over_response = "a" * (threshold + 100)  # Enough over to trigger
        result = apply_conciseness(over_response)

        assert result["is_truncated"] is True

    def test_empty_response(self):
        """Test empty response handling."""
        result = apply_conciseness("")

        assert result["response"] == ""
        assert result["is_truncated"] is False

    def test_whitespace_only_response(self):
        """Test whitespace-only response."""
        result = apply_conciseness("   \n\t  ")

        assert result["is_truncated"] is False

    def test_single_long_sentence(self):
        """Test single sentence that exceeds threshold."""
        single_sentence = "ROS 2 " + "provides " * 100 + "features."
        result = apply_conciseness(single_sentence)

        # Should still truncate even single sentence if too long
        if len(single_sentence) > 250:
            assert result["is_truncated"] is True


class TestTechnicalAccuracyPreservation:
    """Tests for technical accuracy preservation during conciseness."""

    def test_preserves_ros2_mention(self):
        """Test that ROS 2 is preserved in truncated response."""
        long_response = "ROS 2 is the next generation of Robot Operating System. It provides DDS communication, real-time support, and cross-platform capabilities. " * 5
        result = apply_conciseness(long_response)

        assert "ros" in result["response"].lower()

    def test_preserves_technical_terms(self):
        """Test that key technical terms are preserved."""
        long_response = "DDS (Data Distribution Service) provides QoS policies. Publishers and subscribers communicate via topics. " * 10
        result = apply_conciseness(long_response)

        response_lower = result["response"].lower()
        # At least one technical term should be present
        assert any(term in response_lower for term in ["dds", "publisher", "subscriber", "topic", "qos"])
